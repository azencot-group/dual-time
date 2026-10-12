"""Repo-relative path resolution.

Every path in the configs and the evaluators is written relative to the
repository root (the directory containing ``src/``):

    artifacts/models/Qwen3-VL-8B-Instruct      base model
    artifacts/checkpoints/<run>/...            training outputs
    data/stageA/..., data/stageB/...           training corpora
    eval_data/tsrbench, eval_data/chatts_eval  benchmark data
    results/, logs/                            evaluation outputs, launcher logs

`resolve` turns such a string into an absolute path so the code works from
any working directory; absolute paths are returned unchanged and ``~`` is
expanded. `resolve_cfg_paths` applies the rule to a loaded training config.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)

# Config keys whose value is always a filesystem path (resolved even when the
# target does not exist yet, e.g. a fresh output_dir).
_ALWAYS_PATH_KEYS = ("output_dir",)
_PATH_KEY_SUFFIXES = ("path", "_dir", "adapter", "output_dir")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def resolve(p):
    """Absolute path for `p`: unchanged if already absolute, ``~`` expanded,
    otherwise joined onto REPO_ROOT. Non-strings are returned as they are."""
    if not isinstance(p, str) or not p:
        return p
    p = os.path.expanduser(p)
    if os.path.isabs(p):
        return p
    return os.path.join(REPO_ROOT, p)


def _resolve_value(key, value):
    """Resolve one string config value if it denotes a repo-relative path.

    The rule: only keys ending in ``path``, ``_dir``, ``adapter`` or
    ``output_dir`` are considered (this covers ``model_path``, ``output_dir``,
    ``resume_adapter`` and the ``path`` field of ``unified_sources`` entries).
    Such a value is resolved against REPO_ROOT when
      * it is absolute or starts with ``~`` (then `resolve` only normalises it),
      * the key is ``output_dir`` (resolved unconditionally; the trainer
        creates the directory), or
      * the resolved path exists on disk.
    URLs (``scheme://``) are never touched. A Hub id such as
    ``facebook/dinov3-vitl16-pretrain-lvd1689m`` (``dino_model_id``) or
    ``Qwen/Qwen3-VL-8B-Instruct`` in ``model_path`` stays as written: it
    contains a ``/`` but no file of that name exists under the repo.
    """
    if not isinstance(value, str) or not value:
        return value
    if not isinstance(key, str) or not key.endswith(_PATH_KEY_SUFFIXES):
        return value
    if _SCHEME.match(value):
        return value
    if os.path.isabs(value) or value.startswith("~"):
        return resolve(value)
    if key in _ALWAYS_PATH_KEYS:
        return resolve(value)
    cand = resolve(value)
    if os.path.exists(cand):
        return cand
    return value   # a Hub id, or a path that does not exist (reported by the consumer)


def resolve_cfg_paths(obj, _key=None):
    """Recursively apply `_resolve_value` to every string of a loaded config
    (dicts and lists; the rule is documented there)."""
    if isinstance(obj, dict):
        return {k: resolve_cfg_paths(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_cfg_paths(v, _key) for v in obj]
    return _resolve_value(_key, obj)
