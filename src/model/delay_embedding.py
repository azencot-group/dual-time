"""Delay-embedded image rendering for time series.

Each 1-D series is turned into a (H, W) matrix by sliding a fixed-length window across the
series with a step proportional to (length - window) / (W - 1). The window
values are interpolated to fill H rows. The 2-D map is normalised to [0, 1]
using min/max and broadcast across 3 channels.

Output is a **(3, H, W) float32 tensor in [0, 1]** — no uint8 quantisation,
no PIL round-trip. The collators feed it straight to the DINO normaliser
((x - mean) / std) without any /255 step. `delay_tensor_to_pil` is provided
for visualisation paths that still want a PIL image.

Normalisation scope:
  - Default: per-series (uses the current 1-D array's own min/max).
  - With `vmin/vmax` overrides (`channel_stats`, a {field -> {vmin, vmax}}
    JSON given explicitly via `channel_stats_path`): the per-channel min/max
    is fixed across the dataset. Off in every shipped config and evaluation.

For TimeSeriesExam examples that carry multiple series (ts1+ts2), we emit one
delay-embedded tensor per series (paired examples -> 2 tensors).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src import paths

DEFAULT_HW = 256
DEFAULT_EMBED_RATIO = 0.6
DEFAULT_EMBED_LMIN = 48
DEFAULT_EMBED_LMAX = 192


def _to_1d(ts) -> np.ndarray:
    arr = np.asarray(ts, dtype=np.float32).squeeze()
    if arr.ndim == 0:
        arr = arr[None]
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D series, got shape {arr.shape}")
    return arr


def _delay_embed_2d(
    x: np.ndarray,
    height: int,
    width: int,
    embed_ratio: float,
    embed_lmin: int,
    embed_lmax: int,
) -> np.ndarray:
    if x.size == 0:
        x = np.zeros(height, dtype=np.float32)

    length = int(x.size)
    raw_l = int(math.floor(embed_ratio * length))
    l = min(embed_lmax, max(embed_lmin, raw_l))
    l = max(1, min(l, length))

    max_start = max(0, length - l)
    delay = (max_start / float(width - 1)) if width > 1 else 0.0

    out = np.empty((height, width), dtype=np.float32)
    src = np.linspace(0.0, 1.0, num=l, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=height, dtype=np.float32)

    for col in range(width):
        start = int(round(col * delay))
        if start > max_start:
            start = max_start
        window = x[start : start + l]
        if l == height:
            out[:, col] = window
        elif l == 1:
            out[:, col] = window[0]
        else:
            out[:, col] = np.interp(dst, src, window).astype(np.float32)
    return out


def render_single_delay(
    ts,
    base_height: int = DEFAULT_HW,
    base_width: int = DEFAULT_HW,
    embed_ratio: float = DEFAULT_EMBED_RATIO,
    embed_lmin: int = DEFAULT_EMBED_LMIN,
    embed_lmax: int = DEFAULT_EMBED_LMAX,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> torch.Tensor:
    """Render one 1-D series to a delay-embedded RGB tensor in float32 [0, 1].

    Returns a (3, H, W) float32 tensor. No uint8 quantisation, no PIL — the
    collators apply DINO's (x - mean) / std directly.

    If `vmin` and `vmax` are provided, use them for the [0,1] rescale (matches
    the original repo, where per-channel min/max are computed once across the
    whole dataset). Otherwise fall back to per-series min/max.
    """
    x = _to_1d(ts)
    if vmin is None or vmax is None:
        vmin = float(np.nanmin(x)) if np.any(np.isfinite(x)) else 0.0
        vmax = float(np.nanmax(x)) if np.any(np.isfinite(x)) else 1.0
    if vmax > vmin:
        xn = (x - vmin) / (vmax - vmin)
    else:
        xn = np.zeros_like(x)
    xn = np.clip(xn, 0.0, 1.0)

    x2d = _delay_embed_2d(xn, base_height, base_width, embed_ratio, embed_lmin, embed_lmax)
    # (H, W) -> (3, H, W) float32 in [0, 1]
    t = torch.from_numpy(x2d).to(torch.float32)
    return t.unsqueeze(0).expand(3, -1, -1).contiguous()


def render_for_example_delay(
    example: dict,
    channel_stats: Optional[Dict[str, Dict[str, float]]] = None,
    **kwargs,
) -> List[torch.Tensor]:
    """Return one (3, H, W) float32 tensor per time-series in the example.

    Inspects standard TimeSeriesExam fields:
      - `ts`         -> [one tensor]
      - `ts1`,`ts2`  -> [t1, t2]
      - fallback: any field whose value looks like a numeric 1-D series.

    `channel_stats`, when given, maps field name -> {"vmin": float, "vmax": float}
    and overrides the per-series min/max so all examples share a per-field scale.
    """
    ts1 = example.get("ts1")
    ts2 = example.get("ts2")
    ts = example.get("ts")

    named_series: List[Tuple[str, object]] = []
    if ts1 is not None and len(ts1):
        named_series.append(("ts1", ts1))
    if ts2 is not None and len(ts2):
        named_series.append(("ts2", ts2))
    if not named_series and ts is not None and len(ts):
        named_series.append(("ts", ts))

    if not named_series:
        for k, v in example.items():
            if isinstance(v, (list, tuple)) and len(v) > 8 and all(
                isinstance(x, (int, float)) for x in v[:5]
            ):
                named_series.append((k, v))

    images: List[torch.Tensor] = []
    for name, s in named_series:
        if channel_stats and name in channel_stats:
            stats = channel_stats[name]
            images.append(render_single_delay(
                s, vmin=stats["vmin"], vmax=stats["vmax"], **kwargs,
            ))
        else:
            images.append(render_single_delay(s, **kwargs))
    return images


def delay_tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Visualisation helper: (3, H, W) float32 [0, 1] tensor -> PIL.RGB.

    Used by preview / sample-HTML scripts that need a PIL image for `.save()`.
    Loses the same precision the old pipeline did; the training path never
    touches this — it stays in float."""
    if t.ndim == 3 and t.shape[0] == 3:
        arr = t.permute(1, 2, 0).cpu().float().clamp(0.0, 1.0).numpy()
    elif t.ndim == 2:
        arr = t.cpu().float().clamp(0.0, 1.0).numpy()
        arr = np.stack([arr] * 3, axis=-1)
    else:
        raise ValueError(f"unexpected shape: {tuple(t.shape)}")
    return Image.fromarray((arr * 255.0).astype(np.uint8))


if __name__ == "__main__":
    import os
    out = paths.resolve("results/preview_delay")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(3):
        ts = np.cumsum(rng.standard_normal(256)) + np.sin(np.linspace(0, 6 * np.pi, 256))
        t = render_single_delay(ts)
        delay_tensor_to_pil(t).save(f"{out}/delay_{i}.jpg")
    print(f"Wrote delay-embedding previews to {out}")
