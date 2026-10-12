"""The unified-example schema every loader in src/data emits, and the
raw-row loader for pre-built unified JSONL files.

    {
        "system":      str | None,   # optional system prompt
        "user_text":   str,          # user turn; contains literal <ts><ts/>
                                     # markers, one per time series
        "target_text": str,          # assistant turn to train on
        "ts_list":     list[np.ndarray],  # 1-D arrays, order matches markers
        "task":        str,          # "chatts" | "tsexam_perception" | "numerical_corpus" | ...
        "meta":        dict,         # category/idx etc.
        "ts_field_names": list[str], # parallel to ts_list (chart labels / global norm)
        "reasoning_text": str?,      # optional rationale for thinking-format targets
    }

ChatTS rows already carry the `<ts><ts/>` markers inline; the other loaders
synthesise them, one per series, prepended to the question.
"""

from __future__ import annotations

from typing import List

TS_MARKER = "<ts><ts/>"


def load_unified_jsonl(path: str, limit=None) -> List[dict]:
    """Load pre-built unified rows from a JSONL file.

    Each line is already a unified row (system, user_text, target_text,
    optional reasoning_text, ts_list, ts_field_names, task, meta); the series
    are deserialised back to float32 arrays. Handy for custom datasets.
    """
    import json as _json
    import numpy as _np
    out: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = _json.loads(line)
            r["ts_list"] = [_np.asarray(a, dtype=_np.float32) for a in r.get("ts_list", [])]
            out.append(r)
            if limit and len(out) >= int(limit):
                break
    return out
