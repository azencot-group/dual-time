"""Collator for the dual-tower training/eval path.

Each example (see chatts_data.py) carries a `user_text` with literal
`<ts><ts/>` markers, an ordered `ts_list`, and a free-text `target_text`.

For every marker the collator emits TWO vision spans:
  1. an *image* span: a matplotlib chart of the series, encoded by the Qwen
     image processor and consumed by the native Qwen ViT;
  2. a *video* span: a 256x256 delay-embedding image of the same series,
     consumed by the DINOv3 tower (the video stream is repurposed for this).

So a batch carries `pixel_values`/`image_grid_thw` (charts) and
`pixel_values_videos`/`video_grid_thw` (delay images). Labels are masked to
the assistant span; generation mode drops the assistant turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Sequence

import numpy as np
import torch

from src.model.delay_embedding import render_single_delay

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TS_MARKER = "<ts><ts/>"


def _to_chw_normed(img, mean, std) -> torch.Tensor:
    """Take a (3, H, W) float32 tensor in [0, 1] (the delay-embedding output)
    OR a PIL.Image / HxWx3 uint8 array, and return a DINO-normalised
    (3, H, W) float32 tensor."""
    if isinstance(img, torch.Tensor):
        t = img.to(torch.float32)
        if t.ndim != 3 or t.shape[0] != 3:
            raise ValueError(f"expected (3, H, W) tensor, got {tuple(t.shape)}")
        mean_t = torch.as_tensor(mean, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.as_tensor(std, dtype=torch.float32).view(3, 1, 1)
        return (t - mean_t) / std_t
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))


_PLACEHOLDER_NAME_RE = re.compile(r"\w+_s\d+")
_V2_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def _real_name(nm) -> str:
    """A usable series label, or '' for missing/fabricated placeholder names."""
    nm = str(nm) if nm else ""
    return "" if (not nm or _PLACEHOLDER_NAME_RE.fullmatch(nm)) else nm


def _native_dims(tokens: int, px_per_tok: int = 1024, round_to: int = 32,
                 aspect: float = 2.3):
    """Figure (w, h) in px for a token budget: dims are multiples of the ViT
    patch*merge size so the processor ingests the render with NO resampling."""
    px = tokens * px_per_tok
    w = max(round_to, int(round((px * aspect) ** 0.5 / round_to) * round_to))
    h = max(round_to, (px // w) // round_to * round_to)
    return w, h


def _v2_axes_style(ax, w: int):
    from matplotlib.ticker import MaxNLocator, AutoMinorLocator
    dense = w >= 320
    ax.grid(True, which="major", alpha=0.35)
    ax.xaxis.set_major_locator(MaxNLocator(10 if dense else 5))
    ax.yaxis.set_major_locator(MaxNLocator(8 if dense else 4))
    if dense:
        ax.grid(True, which="minor", alpha=0.15)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _v2_fig_to_pil(fig):
    import io
    from PIL import Image
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=100)
    plt.close("all")
    buf.seek(0)
    return Image.open(buf).convert("RGB").copy()


def _chart_v2_single(series, w: int, h: int):
    """One v2 chart, no in-figure labels: series names live in the TEXT
    ('{name}: <ts><ts/>', see _label_text_markers)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.plot(np.asarray(series, dtype=np.float32).ravel(), lw=1.8, color=_V2_COLORS[0])
    _v2_axes_style(ax, w)
    fig.tight_layout()
    return _v2_fig_to_pil(fig)


def _names_are_choices(fnames) -> bool:
    """True when ANY series is an answer candidate (series-as-choices: names
    'choice_A'.., 'Choice A: x(t)', or history + candidates). Such examples
    must never be merged into one overlay chart: the model has to compare
    separate plots to pick one."""
    if not fnames:
        return False
    return any(str(n).lower().replace(" ", "_").startswith("choice") for n in fnames)


def _label_text_markers(text: str, fnames) -> str:
    """v2: put series names in the TEXT. Each marker with a real field name
    becomes '{name}: <ts><ts/>'; adjacent labeled markers go one per line."""
    parts = text.split(TS_MARKER)
    n = len(parts) - 1
    out = parts[0]
    for i in range(n):
        nm = _real_name(fnames[i]) if fnames and i < len(fnames) else ""
        seg = (f"{nm}: " if nm else "") + TS_MARKER
        if nm and i < n - 1 and parts[i + 1] == "":
            seg += "\n"
        out += seg + parts[i + 1]
    return out


def _series_chart(series, title: str):
    """v1 renderer: one series as a 6x2 titled matplotlib chart (ChatTS style)."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.figure(figsize=(6, 2))
    plt.plot(np.asarray(series, dtype=np.float32).ravel(), linewidth=2, color="blue")
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    plt.close("all")
    buf.seek(0)
    return Image.open(buf).convert("RGB").copy()


@dataclass
class UnifiedDualCollator:
    """Collator for the dual vision-tower setup (module docstring).

    `channel_stats`: optional {field_name -> {"vmin", "vmax"}} used to
    globally normalise the DELAY embedding (the chart goes through Qwen's
    processor unchanged). Examples must then carry `ts_field_names`
    (parallel to `ts_list`). Without it every series is instance-normalised.
    """
    processor: Any
    max_length: int = 4096
    delay_image_size: int = 256
    delay_patch_size: int = 16
    delay_spatial_merge_size: int = 2
    image_token: str = "<|image_pad|>"
    video_token: str = "<|video_pad|>"
    vision_start: str = "<|vision_start|>"
    vision_end: str = "<|vision_end|>"
    image_mean: Sequence[float] = field(default_factory=lambda: IMAGENET_MEAN)
    image_std: Sequence[float] = field(default_factory=lambda: IMAGENET_STD)
    channel_stats: Any = None
    # Chart token budgeting. Each <ts> emits a chart (native ViT) + a delay
    # image (DINO). Charts are sized PER EXAMPLE so that charts + delay images
    # always fit in max_length - text_reserve: few-series examples keep
    # full-size charts, many-series examples get smaller ones.
    text_reserve: int = 1536
    min_chart_tokens: int = 16
    max_chart_tokens: int = 114        # v1 per-chart cap
    max_chart_tokens_v2: int = 240     # v2 per-chart cap (-> 736x320 px = 230 tokens)
    chart_labels: bool = False         # v1 only: title charts with the field name
    # Chart renderer. "v2" (default): rendered at the exact ViT-native pixel
    # size (no resample), dense grid with tick values, one chart per series,
    # and series names injected into the TEXT as '{name}: <ts><ts/>'.
    # "v1": one 6x2 titled chart per series, resampled by the processor.
    chart_style: str = "v2"

    def __post_init__(self):
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        self.merge_size = int(getattr(self.image_processor, "merge_size", 2))
        self.chart_patch = int(getattr(self.image_processor, "patch_size", 16))
        # pixels per merged token = patch_size^2 * merge_size^2
        self.px_per_tok = (self.chart_patch * self.merge_size) ** 2
        g = self.delay_image_size // self.delay_patch_size
        if g % self.delay_spatial_merge_size != 0:
            raise ValueError("delay grid not divisible by spatial_merge_size")
        self.delay_grid = g
        self.delay_tokens = (g * g) // (self.delay_spatial_merge_size ** 2)
        if self.chart_style not in ("v1", "v2"):
            raise ValueError(f"unknown chart_style: {self.chart_style!r}")

    def _delay_pixels(self, ts, field_name: str = None) -> torch.Tensor:
        vmin, vmax = None, None
        if self.channel_stats and field_name and field_name in self.channel_stats:
            s = self.channel_stats[field_name]
            vmin, vmax = float(s["vmin"]), float(s["vmax"])
        img = render_single_delay(ts, base_height=self.delay_image_size,
                                  base_width=self.delay_image_size,
                                  vmin=vmin, vmax=vmax)
        return _to_chw_normed(img, self.image_mean, self.image_std)

    def _chart_token_cap(self, n_series: int) -> int:
        """v1 per-chart token cap so charts + delay images fit the budget."""
        if n_series <= 0:
            return self.max_chart_tokens
        budget = max(self.max_length - self.text_reserve,
                     n_series * (self.min_chart_tokens + self.delay_tokens))
        cap = budget // n_series - self.delay_tokens
        return int(max(self.min_chart_tokens, min(self.max_chart_tokens, cap)))

    def _chart_features_v2(self, ts_list):
        """v2 renderer: native-resolution figures, one chart per series."""
        n = len(ts_list)
        rnd = self.chart_patch * self.merge_size
        budget = max(self.max_length - self.text_reserve,
                     n * (self.min_chart_tokens + self.delay_tokens))
        cap = int(max(self.min_chart_tokens,
                      min(self.max_chart_tokens_v2, budget // n - self.delay_tokens)))
        w, h = _native_dims(cap, self.px_per_tok, rnd)
        charts = [_chart_v2_single(ts, w, h) for ts in ts_list]
        enc = self.image_processor(images=charts, return_tensors="pt",
                                   max_pixels=cap * self.px_per_tok,
                                   min_pixels=self.min_chart_tokens * self.px_per_tok)
        grid = enc["image_grid_thw"]
        tok = (grid.prod(-1) // (self.merge_size ** 2)).tolist()
        return enc["pixel_values"], grid, tok

    def _chart_features(self, ts_list, fnames=None):
        """Render one chart per series and run them through the Qwen image
        processor. Returns (pixel_values, image_grid_thw, tokens_per_chart)."""
        if not ts_list:
            return None, None, []
        if self.chart_style == "v2":
            return self._chart_features_v2(ts_list)

        def _title(i):
            nm = str(fnames[i]) if fnames and i < len(fnames) and fnames[i] else ""
            if self.chart_labels and nm and not _PLACEHOLDER_NAME_RE.fullmatch(nm):
                return nm
            return f"series {i + 1}"
        charts = [_series_chart(ts, _title(i)) for i, ts in enumerate(ts_list)]
        cap = self._chart_token_cap(len(ts_list))
        kw = {}
        if cap < self.max_chart_tokens:
            kw = {"max_pixels": cap * self.px_per_tok,
                  "min_pixels": self.min_chart_tokens * self.px_per_tok}
        enc = self.image_processor(images=charts, return_tensors="pt", **kw)
        grid = enc["image_grid_thw"]
        tok = (grid.prod(-1) // (self.merge_size ** 2)).tolist()
        return enc["pixel_values"], grid, tok

    def _expand_markers(self, text: str, chart_tok: List[int]) -> str:
        """Replace each `<ts><ts/>` with an image span then a video span."""
        parts = text.split(TS_MARKER)
        found = len(parts) - 1
        if found != len(chart_tok):
            raise ValueError(f"example has {len(chart_tok)} series but {found} markers")
        vid_span = self.vision_start + self.video_token * self.delay_tokens + self.vision_end
        out = parts[0]
        for i in range(found):
            img_span = self.vision_start + self.image_token * chart_tok[i] + self.vision_end
            out += img_span + vid_span + parts[i + 1]
        return out

    def _build_messages(self, ex: dict, with_target: bool) -> list:
        msgs = []
        if ex.get("system"):
            msgs.append({"role": "system", "content": [{"type": "text", "text": ex["system"]}]})
        msgs.append({"role": "user", "content": [{"type": "text", "text": ex["user_text"]}]})
        if with_target:
            asst = {"role": "assistant",
                    "content": [{"type": "text", "text": ex["target_text"]}]}
            # Thinking-format target: the rationale is rendered into the
            # model's <think>..</think> block via `reasoning_content`; the
            # answer stays the bare `content`. Absent -> a plain turn with the
            # empty (pre-closed) think block.
            if ex.get("reasoning_text"):
                asst["reasoning_content"] = ex["reasoning_text"]
            msgs.append(asst)
        return msgs

    def __call__(self, batch: List[dict], for_generation: bool = False):
        chart_pv, chart_grid, delay_imgs = [], [], []
        per_ex_tok = []
        for ex in batch:
            pv, grid, tok = self._chart_features(ex["ts_list"], ex.get("ts_field_names"))
            if pv is not None:
                chart_pv.append(pv)
                chart_grid.append(grid)
            per_ex_tok.append(tok)
            fnames = ex.get("ts_field_names") or [None] * len(ex["ts_list"])
            for ts, fname in zip(ex["ts_list"], fnames):
                delay_imgs.append(self._delay_pixels(ts, field_name=fname))

        full_texts, prompt_texts = [], []
        for ex, tok in zip(batch, per_ex_tok):
            if self.chart_style == "v2":
                ex = dict(ex)
                ex["user_text"] = _label_text_markers(
                    ex["user_text"], ex.get("ts_field_names"))
            # enable_thinking is passed EXPLICITLY per example: the template
            # defaults it to True, which would open <think> for every turn.
            #   reasoning     -> True : prompt opens '<think>\n' (label starts inside)
            #   non-reasoning -> False: prompt carries the empty, pre-closed
            #                           think block so only the answer is learned.
            # Generation examples carry no reasoning_text, so at evaluation
            # thinking is always off.
            think = bool(ex.get("reasoning_text"))
            prompt = self.processor.apply_chat_template(
                self._build_messages(ex, with_target=False),
                tokenize=False, add_generation_prompt=True,
                enable_thinking=think)
            prompt_texts.append(self._expand_markers(prompt, tok))
            if not for_generation:
                full = self.processor.apply_chat_template(
                    self._build_messages(ex, with_target=True),
                    tokenize=False, add_generation_prompt=False)
                full_texts.append(self._expand_markers(full, tok))

        texts = prompt_texts if for_generation else full_texts
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        if for_generation and int(enc["attention_mask"].sum(1).max()) >= self.max_length:
            # A silently truncated prompt loses its tail (options, "Answer:",
            # the assistant header) and the model just continues the text;
            # fail loudly so the budget gets raised (--max-length).
            raise ValueError(f"generation prompt truncated at max_length={self.max_length}; "
                             "raise --max-length (e.g. 16384)")

        if not for_generation:
            # Mask everything up to the token-level common prefix of prompt vs
            # full. Exact even when the prompt/target boundary merges into one
            # token (e.g. `<think>\n` + rationale). Assumes right padding.
            labels = enc["input_ids"].clone()
            for i, pt in enumerate(prompt_texts):
                prompt_ids = self.tokenizer(pt, add_special_tokens=False).input_ids
                full_ids = enc["input_ids"][i].tolist()
                cp = 0
                for a, b in zip(prompt_ids, full_ids):
                    if a != b:
                        break
                    cp += 1
                labels[i, :cp] = -100
            labels[enc["attention_mask"] == 0] = -100
            enc["labels"] = labels

        if chart_pv:
            enc["pixel_values"] = torch.cat(chart_pv, dim=0)
            enc["image_grid_thw"] = torch.cat(chart_grid, dim=0)
        if delay_imgs:
            enc["pixel_values_videos"] = torch.stack(delay_imgs, dim=0)
            enc["video_grid_thw"] = torch.tensor(
                [[1, self.delay_grid, self.delay_grid]] * len(delay_imgs), dtype=torch.long)

        # A truncated vision span would silently misalign features against
        # placeholder tokens; fail loudly instead.
        img_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        vid_id = self.tokenizer.convert_tokens_to_ids(self.video_token)
        n_img_pads = int((enc["input_ids"] == img_id).sum())
        n_vid_pads = int((enc["input_ids"] == vid_id).sum())
        want_img = int(sum(sum(t) for t in per_ex_tok))
        want_vid = len(delay_imgs) * self.delay_tokens
        if n_img_pads != want_img or n_vid_pads != want_vid:
            raise ValueError(
                f"vision span truncated: image_pads {n_img_pads}/{want_img}, "
                f"video_pads {n_vid_pads}/{want_vid}; an example exceeds "
                f"max_length={self.max_length}; raise max_length or shrink charts")
        return dict(enc)
