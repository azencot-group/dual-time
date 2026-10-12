"""TimeSeriesExam loaders: the TSExam perception set (`tsexam_perception`,
multiple choice) and the numerical corpus (`numerical_corpus`, open-ended
scalar answers) - both from the same generator family.
"""

from __future__ import annotations

import json
import random
from typing import List, Optional

import numpy as np

from .common import TS_MARKER
from .mcq import _shuffle_mcq, resolve_letter
from .prompts import SYSTEM as TSEXAM_SYSTEM, user_text as tsexam_user_text


def tsexam_perception_rows_to_unified(rows, indices=None, shuffle_options=False,
                           shuffle_seed=0) -> List[dict]:
    """Convert a list of TimeSeriesExam-schema rows into unified examples.

    Each example becomes: N `<ts><ts/>` markers (one per series) followed by the
    question + options; target is the gold letter. `indices`, when given,
    selects a subset and is recorded in meta['idx'].

    `shuffle_options=True` applies a deterministic per-row permutation to the
    option order (seeded by `shuffle_seed` + row index, stable across runs and
    ranks) and remaps the gold letter, so the model can't learn letter priors.

    Each example also carries `ts_field_names` (parallel to `ts_list`) so a
    downstream collator with per-field global-norm stats can look up the
    right (vmin, vmax) per series.
    """
    if indices is None:
        indices = range(len(rows))
    out: List[dict] = []
    for i in indices:
        r = rows[i]
        try:
            options = list(r["options"])
            gold = resolve_letter(r["answer"], options)
        except Exception:
            continue
        option_perm = None
        if shuffle_options and len(options) > 1:
            options, gold, option_perm = _shuffle_mcq(
                options, gold, random.Random(f"tsexam:{shuffle_seed}:{i}"))
        # Collect series in a fixed order.
        ts_list = []
        ts_field_names: List[str] = []
        for k in ("ts1", "ts2", "ts"):
            v = r.get(k)
            if v is not None and len(v):
                ts_list.append(np.asarray(v, dtype=np.float32).ravel())
                ts_field_names.append(k)
        if not ts_list:
            continue
        markers = "".join(TS_MARKER for _ in ts_list)
        q = tsexam_user_text(str(r["question"]), options)
        user_text = f"{markers}\n{q}"
        out.append({
            "system": TSEXAM_SYSTEM,
            "user_text": user_text,
            "target_text": gold,
            "ts_list": ts_list,
            "ts_field_names": ts_field_names,
            "task": "tsexam_perception",
            "meta": {
                "idx": i,
                "category": str(r.get("category", "")),
                "subcategory": str(r.get("subcategory", "")),
                "difficulty": str(r.get("difficulty", "")),
                "gold_letter": gold,
                "n_options": len(options),
                "option_perm": option_perm,
            },
        })
    return out


def load_tsexam_perception(json_path: str, limit: Optional[int] = None,
                        shuffle_options: bool = False,
                        shuffle_seed: int = 0) -> List[dict]:
    """Load a TimeSeriesExam JSON file as unified examples."""
    with open(json_path) as f:
        rows = json.load(f)
    idx = range(len(rows)) if limit is None else range(min(limit, len(rows)))
    return tsexam_perception_rows_to_unified(rows, indices=idx,
                                  shuffle_options=shuffle_options,
                                  shuffle_seed=shuffle_seed)


TSEXAM_NUMERIC_SYSTEM = (
    "You are a careful time-series analyst. The user shows you one or more "
    "time series and asks a question whose answer is a single number. Answer "
    "with ONLY that number, rounded to 2 decimal places, with no units, "
    "explanation, or extra text."
)


def numerical_corpus_rows_to_unified(rows, indices=None) -> List[dict]:
    """Convert numerical-open-ended TimeSeriesExam rows into unified examples.

    The schema is the same as the standard TSExam JSON except `answer` is a
    float, `options` is empty, and `question_type == 'numeric'`. Each row also
    carries a `format_hint` instructing the model how to format the answer.
    """
    if indices is None:
        indices = range(len(rows))
    out: List[dict] = []
    for i in indices:
        r = rows[i]
        try:
            ans = float(r["answer"])
        except (KeyError, TypeError, ValueError):
            continue
        ts_list = []
        ts_field_names: List[str] = []
        for k in ("ts1", "ts2", "ts"):
            v = r.get(k)
            if v is not None and len(v):
                ts_list.append(np.asarray(v, dtype=np.float32).ravel())
                ts_field_names.append(k)
        if not ts_list:
            continue
        markers = "".join(TS_MARKER for _ in ts_list)
        fmt_hint = str(r.get("format_hint", "")).strip()
        user_text = f"{markers}\n{r['question']}"
        if fmt_hint:
            user_text += f"\n\n{fmt_hint}"
        target_text = f"{ans:.2f}"
        out.append({
            "system": TSEXAM_NUMERIC_SYSTEM,
            "user_text": user_text,
            "target_text": target_text,
            "ts_list": ts_list,
            "ts_field_names": ts_field_names,
            "task": "numerical_corpus",
            "meta": {
                "idx": i,
                "id": r.get("id"),
                "tid": r.get("tid"),
                "category": str(r.get("category", "")),
                "subcategory": str(r.get("subcategory", "")),
                "difficulty": str(r.get("difficulty", "")),
                "gold_value": ans,
            },
        })
    return out


def load_numerical_corpus(json_path: str, limit: Optional[int] = None) -> List[dict]:
    """Load a numerical-open-ended TimeSeriesExam JSON as unified examples."""
    with open(json_path) as f:
        rows = json.load(f)
    idx = range(len(rows)) if limit is None else range(min(limit, len(rows)))
    return numerical_corpus_rows_to_unified(rows, indices=idx)
