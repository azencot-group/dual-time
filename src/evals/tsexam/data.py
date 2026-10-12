"""The TimeSeriesExam1 benchmark (AutonLab/TimeSeriesExam1, 746 MCQ), pinned to v1.1.

`load_raw` returns the rows (Hub revision `BENCHMARK_REVISION`, or a local JSON of
the same schema) and `benchmark_rows_to_examples` turns them into the unified
example dicts the collator renders: one `<ts><ts/>` marker per series, the
TSExam letter prompt, the gold letter and category/subcategory/difficulty meta.
Evaluation-only; the training-side loaders live in src/datasets_utils/.
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

import numpy as np
from datasets import load_dataset

from src.datasets_utils.common import TS_MARKER
from src.datasets_utils.mcq import resolve_letter
from src.datasets_utils.prompts import SYSTEM as TSEXAM_SYSTEM, user_text as tsexam_user_text

DATASET_NAME = "AutonLab/TimeSeriesExam1"
# v1.1 (dataset README changelog, 2025-03-12); v1.0 was 9f23771c. `main` points here.
BENCHMARK_REVISION = "628f686757bb484d9b608b6855fb41f6a302ecda"
DEFAULT_CACHE = None   # the `datasets` library's own cache (HF_HOME if set)


def load_raw(cache_dir: Optional[str] = DEFAULT_CACHE, json_path: Optional[str] = None):
    """Return a list of TimeSeriesExam rows.

    With `json_path` the rows come from that local JSON file (schema: question,
    options, answer, ts/ts1/ts2, category, subcategory, difficulty); otherwise
    every split of the HF dataset at `BENCHMARK_REVISION` is concatenated
    (`cache_dir=None` = the default Hugging Face datasets cache).
    """
    if json_path:
        with open(json_path) as f:
            return list(json.load(f))
    ds = load_dataset(DATASET_NAME, cache_dir=cache_dir, revision=BENCHMARK_REVISION)
    rows = []
    for split_name in ds.keys():
        rows.extend(ds[split_name])
    return rows


def benchmark_rows_to_examples(rows, indices=None) -> List[dict]:
    """Benchmark rows -> unified examples for the collator.

    Each example is N `<ts><ts/>` markers (one per series, ts1/ts2/ts order)
    followed by the TSExam letter prompt (question + options); `target_text` is
    the gold letter, `meta` carries idx, category, subcategory, difficulty,
    gold_letter and n_options. Options are never shuffled at evaluation.
    `indices` selects a subset and is recorded in meta['idx'].
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
        user_text = f"{markers}\n{tsexam_user_text(str(r['question']), options)}"
        out.append({
            "system": TSEXAM_SYSTEM,
            "user_text": user_text,
            "target_text": gold,
            "ts_list": ts_list,
            "ts_field_names": ts_field_names,
            "task": "tsexam",
            "meta": {
                "idx": i,
                "category": str(r.get("category", "")),
                "subcategory": str(r.get("subcategory", "")),
                "difficulty": str(r.get("difficulty", "")),
                "gold_letter": gold,
                "n_options": len(options),
            },
        })
    return out


if __name__ == "__main__":  # quick check: every gold answer must resolve to a letter
    p = argparse.ArgumentParser()
    p.add_argument("--json-path", default=None)
    args = p.parse_args()
    rows = load_raw(json_path=args.json_path)
    print(f"[dataset] loaded {len(rows)} rows; columns={list(rows[0].keys())[:20]}...")
    bad = []
    for i, r in enumerate(rows):
        try:
            resolve_letter(r["answer"], list(r["options"]))
        except Exception as e:
            bad.append((i, str(e)[:120]))
    print(f"[dataset] gold-letter resolution: {len(rows)-len(bad)}/{len(rows)} OK; {len(bad)} failures")
    for i, msg in bad[:5]:
        print(f"  row {i}: {msg}")
    assert not bad, "gold-letter resolution failed on some rows"
