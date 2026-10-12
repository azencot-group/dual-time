"""Loaders for the Stage A caption sources.

Reads the structural captions (data/stageA/structural_captions.json,
TSExam generator) and the ChatTS-style caption JSONL, converting each row into the project's
unified-example schema (same shape used by tsexam_perception, numerical_corpus and chatts;
see src/datasets_utils/common.py for the schema docstring).
"""

from __future__ import annotations

import json
from typing import List, Optional

import numpy as np


TS_MARKER = "<ts><ts/>"


# Lexicon-constrained system prompt: the model is trained to use the same
# phrasing as the dataset's caption renderer (utils_caption.render).
CAPTION_SYSTEM = (
    "You are a precise time-series caption assistant. Describe the given series "
    "in one short factual sentence using the same lexicon the dataset uses: "
    "name the component(s) (constant, sine wave, square wave, sawtooth wave, "
    "linear trend, exponential trend, log trend, gaussian white noise, red noise, "
    "autoregressive process, moving average process), state period and amplitude "
    "for waves (period ~X, amplitude ~Y) and value for constants (at ~Z), give "
    "trend direction (upward/downward/flat), use 'a piecewise series: ... , then "
    "... ' for concatenation, ' plus ' for additive, ' multiplied by ' for "
    "multiplicative; append 'containing a <name> anomaly' for any anomaly and "
    "'with gaussian white noise' (or other noise process) at the end if noise was "
    "applied. End with a period."
)

CAPTION_USER_PROMPT = "Describe this time series in one short factual sentence."


def caption_rows_to_unified(rows, indices=None) -> List[dict]:
    """Convert captioned TimeSeriesExam rows into unified examples.

    Only rows that carry both `caption` and `ts` are kept (the captioning
    branch only exists on single-TS samples). The ground-truth `attributes`
    block is preserved in `meta`.
    """
    if indices is None:
        indices = range(len(rows))
    out: List[dict] = []
    for i in indices:
        r = rows[i]
        cap = r.get("caption")
        ts  = r.get("ts")
        if not cap or ts is None or len(ts) == 0:
            continue
        ts_arr = np.asarray(ts, dtype=np.float32).ravel()
        out.append({
            "system": CAPTION_SYSTEM,
            "user_text": f"{TS_MARKER}\n{CAPTION_USER_PROMPT}",
            "target_text": str(cap).strip(),
            "ts_list": [ts_arr],
            "ts_field_names": ["ts"],
            "task": "structural_captions",
            "meta": {
                "idx": i,
                "id": r.get("id"),
                "tid": r.get("tid"),
                "category": str(r.get("category", "")),
                "subcategory": str(r.get("subcategory", "")),
                "difficulty": str(r.get("difficulty", "")),
                "attributes": r.get("attributes", {}),
            },
        })
    return out


def load_structural_captions(json_path: str, limit: Optional[int] = None) -> List[dict]:
    """Load a TimeSeriesExam captioning JSON as unified examples."""
    with open(json_path) as f:
        rows = json.load(f)
    idx = range(len(rows)) if limit is None else range(min(limit, len(rows)))
    return caption_rows_to_unified(rows, indices=idx)


# Generic system prompt for free-form caption JSONL sets (e.g. the
# chatts_align_256_tsexam_captions artifacts). Unlike CAPTION_SYSTEM above, it
# is NOT lexicon-constrained — these captions are free-form prose.
CAPTION_JSONL_SYSTEM = (
    "You are a time-series captioning assistant. Look at the given series and "
    "describe it concisely and factually: its overall trend, notable changes, "
    "peaks/dips, seasonality and noise."
)
CAPTION_JSONL_USER_PROMPT = "Describe this time series."


def load_chatts_style_captions(path: str, limit: Optional[int] = None,
                       target_field: str = "caption") -> List[dict]:
    """Load a JSONL caption set as unified examples (one per line).

    Each line must carry `ts` (1-D series) and the chosen `target_field`
    (default "caption"; "answer" is the other option in the
    chatts_align_256_tsexam_captions artifacts). Mirrors the unified schema
    used by the other loaders; series use the global-norm "ts" channel-stats
    key (these are normal-range 256-step series, like chatts/tsexam).
    """
    out: List[dict] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and len(out) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ts = r.get("ts")
            tgt = r.get(target_field)
            if ts is None or len(ts) < 2 or not tgt:
                continue
            ts_arr = np.asarray(ts, dtype=np.float32).ravel()
            q = (r.get("question") or "").strip()
            user_text = (f"{TS_MARKER}\n{q}\n{CAPTION_JSONL_USER_PROMPT}"
                         if q else f"{TS_MARKER}\n{CAPTION_JSONL_USER_PROMPT}")
            out.append({
                "system": CAPTION_JSONL_SYSTEM,
                "user_text": user_text,
                "target_text": str(tgt).strip(),
                "ts_list": [ts_arr],
                "ts_field_names": ["ts"],
                "task": "chatts_style_captions",
                "meta": {
                    "idx": i,
                    "id": r.get("id"),
                    "tid": r.get("tid"),
                    "subcategory": str(r.get("subcategory", "")),
                    "difficulty": str(r.get("difficulty", "")),
                    "target_field": target_field,
                },
            })
    return out
