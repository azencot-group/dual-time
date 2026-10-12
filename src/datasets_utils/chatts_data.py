"""ChatTS training subsets (ChatTSRepo/ChatTS-Training-Dataset: align_256 /
sft / ift / dev) as unified examples; the rows already carry one `<ts><ts/>`
marker per series inline. Schema: src/datasets_utils/common.py.
"""

from __future__ import annotations

import json
from typing import List, Optional

import numpy as np

from .common import TS_MARKER


def _chatts_path(subset: str) -> str:
    """Resolve a ChatTS subset name to a local jsonl path (downloads + caches)."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        "ChatTSRepo/ChatTS-Training-Dataset",
        f"{subset}/train.jsonl",
        repo_type="dataset",
    )


def load_chatts(subset: str, limit: Optional[int] = None) -> List[dict]:
    """Load a ChatTS subset (align_256 / sft / ift / dev) as unified examples."""
    path = _chatts_path(subset)
    out: List[dict] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            ts = r["timeseries"]
            # timeseries is [C, T]; channels may have differing lengths in sft.
            ts_list = [np.asarray(ch, dtype=np.float32).ravel() for ch in ts]
            user_text = str(r["input"])
            n_marker = user_text.count(TS_MARKER)
            if n_marker != len(ts_list):
                # Skip malformed rows rather than crash the whole run.
                continue
            # ChatTS rows have multi-channel univariate / multivariate data; the
            # individual channels don't carry a TSExam-style "ts/ts1/ts2"
            # field name, so we default each entry to "ts" — useful only when
            # the downstream collator's channel_stats dict has a "ts" entry.
            out.append({
                "system": None,
                "user_text": user_text,
                "target_text": str(r["output"]),
                "ts_list": ts_list,
                "ts_field_names": ["ts"] * len(ts_list),
                "task": "chatts",
                "meta": {"subset": subset, "idx": i},
            })
    return out


if __name__ == "__main__":  # quick check
    ex = load_chatts("ift", limit=2)
    for e in ex:
        print("task:", e["task"], "n_ts:", len(e["ts_list"]),
              "markers:", e["user_text"].count(TS_MARKER))
        print("  user_text head:", e["user_text"][:120])
        print("  target head:", e["target_text"][:120])
