"""Loader for CaTS-Bench captioning as a unified training source.

CaTS-Bench (HF `mhfisher/CaTSBench`, MIT) is a real-world, cross-domain
time-series captioning/QA benchmark. This module adds its **captioning** signal:
for each train series, the model is shown the (short, univariate, real-world)
series plus its metadata and is trained to generate the dataset's synthetic
gold caption.

Distribution is zip archives (not HF-parquet), so we `hf_hub_download` the
zip once (cached) and read the per-series files directly out of the archive:
    train_data/time series/<ts_name>.txt        (one value per line; 5-120 pts)
    train_data/synth_gt_captions/<ts_name>.txt   (gold caption; multi-sentence)
    train_data/metadata/<ts_name>.json           (attribute/country/years/stats)
(time series and caption share the same <ts_name> basename across folders.)

Emits the project's unified-example schema (same shape as src/datasets_utils/caption_data.py
and the other loaders in src/datasets_utils/). Dispatched from
`load_unified_sources` via `{type: catsbench, split: train}`.

Real-world values span very different domain scales, so we tag the series with
`ts_field_names=["catsbench"]` — a key that is intentionally ABSENT from the
channel_stats files, which makes the collator fall back to per-series min/max
normalization (src/datasets_utils/unified_collator.py:_delay_pixels). No channel_stats change.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import List, Optional

import numpy as np

from .caption_data import TS_MARKER

HF_REPO = "mhfisher/CaTSBench"
_SPLIT_ZIP = {"train": "train_data.zip", "test": "test_data.zip"}

CATS_SYSTEM = (
    "You are a time-series analyst. Given a real-world time series (shown as a "
    "chart) and its metadata, write a concise, factual caption describing the "
    "series: its overall trend, notable changes, peaks/dips or anomalies, and "
    "how it compares to its historical norms. Ground the description in the "
    "provided metadata (what is measured, where, the time range, and the "
    "historical statistics)."
)
CATS_USER_PROMPT = "Describe this time series."


def _fmt_num(v) -> str:
    """Compact number formatting (drop trailing zeros, keep ints clean)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f):
        return str(int(f))
    return f"{f:.4g}"


def _context_block(md: dict) -> str:
    """Render the per-series metadata into a compact context the caption needs.

    The gold captions reference the attribute, location, time range, and how
    the series compares to historical/its-own statistics, so those facts must
    be in the prompt for the task to be well-posed.
    """
    if not md:
        return ""
    g = md.get
    parts = []
    if g("attribute"):
        parts.append(f"Attribute: {g('attribute')}")
    if g("country"):
        parts.append(f"Location: {g('country')}")
    sy, ey = g("start year of this series"), g("end year of this series")
    if sy is not None and ey is not None:
        freq = g("sampling frequency")
        parts.append(f"Period: {sy}-{ey}" + (f" ({freq})" if freq else ""))
    hmin, hmean, hmax = g("historical min"), g("historical mean"), g("historical max")
    if hmean is not None:
        parts.append("Historical min/mean/max: "
                     f"{_fmt_num(hmin)}/{_fmt_num(hmean)}/{_fmt_num(hmax)}")
    smin = g("minimum of this specific series")
    smean = g("mean of this specific series")
    smax = g("maximum of this specific series")
    if smean is not None:
        parts.append("This series min/mean/max: "
                     f"{_fmt_num(smin)}/{_fmt_num(smean)}/{_fmt_num(smax)}")
    if g("metrics info"):
        parts.append(f"Info: {g('metrics info')}")
    return "\n".join(parts)


def load_catsbench(split: str = "train", limit: Optional[int] = None) -> List[dict]:
    """Load CaTS-Bench captioning as unified examples (one per train series)."""
    from huggingface_hub import hf_hub_download

    if split not in _SPLIT_ZIP:
        raise ValueError(f"catsbench: unknown split {split!r} "
                         f"(expected one of {list(_SPLIT_ZIP)})")
    zip_path = hf_hub_download(HF_REPO, _SPLIT_ZIP[split], repo_type="dataset")
    top = f"{split}_data"

    zf = zipfile.ZipFile(zip_path)
    names = set(zf.namelist())
    ts_files = sorted(n for n in names
                      if n.startswith(f"{top}/time series/") and n.endswith(".txt"))

    out: List[dict] = []
    for tf in ts_files:
        ts_name = tf[len(f"{top}/time series/"):-4]   # e.g. "agriculture_0_train"
        cap_path = f"{top}/synth_gt_captions/{ts_name}.txt"
        md_path = f"{top}/metadata/{ts_name}.json"
        if cap_path not in names:
            continue
        # time series: one float per line
        raw = zf.read(tf).decode("utf-8", "replace")
        vals = [float(x) for x in raw.split() if x.strip()]
        if len(vals) < 2:
            continue
        ts_arr = np.asarray(vals, dtype=np.float32).ravel()
        caption = zf.read(cap_path).decode("utf-8", "replace").strip()
        if not caption:
            continue
        md = {}
        if md_path in names:
            try:
                md = json.loads(zf.read(md_path))
            except (json.JSONDecodeError, ValueError):
                md = {}
        ctx = _context_block(md)
        user_text = f"{TS_MARKER}\n{ctx}\n{CATS_USER_PROMPT}" if ctx \
            else f"{TS_MARKER}\n{CATS_USER_PROMPT}"
        out.append({
            "system": CATS_SYSTEM,
            "user_text": user_text,
            "target_text": caption,
            "ts_list": [ts_arr],
            "ts_field_names": ["catsbench"],
            "task": "catsbench",
            "meta": {
                "ts_name": ts_name,
                "domain": ts_name.split("_")[0],
                "attribute": md.get("attribute"),
                "country": md.get("country"),
                "n_points": len(vals),
            },
        })
        if limit is not None and len(out) >= limit:
            break
    return out
