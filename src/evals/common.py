"""Pieces shared by the three evaluators (src.evals.{tsexam,tsrbench,chatts}).

* `load_model` / `init_distributed`: the dual-tower model loader for
  evaluation (base + Stage A prior merged first, then the Stage B adapter)
  and the torchrun-aware init (items are sharded across ranks and gathered
  on rank 0 by each evaluator).
* `add_dual_model_args`, `build_dual_collator`, `greedy_cfg`, `run_one`: the
  shared CLI flags, collator setup and greedy decoding (every sampling field
  pinned so the model's generation_config.json cannot leak through).
* `provenance`: the block every result JSON carries (checkpoints, renderer,
  budgets, the full parsed CLI namespace, code revision).
* `extract_letter` / `extract_letter_from_json` / `per_group_accuracy` /
  `strip_think` / `enable_determinism`: answer parsing and the bit-exact
  kernel switch.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import torch

from src import paths

LETTERS = "ABCDEFGH"
DEFAULT_MODEL = paths.resolve("artifacts/models/Qwen3-VL-8B-Instruct")
DEFAULT_DINO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
_REPO_ROOT = Path(paths.REPO_ROOT)


# --------------------------------------------------------------------------
# Model loading and distributed init
# --------------------------------------------------------------------------

def init_distributed():
    """Initialize torch.distributed when launched via torchrun; else single-process.

    Each rank pins itself to its own physical GPU via CUDA_VISIBLE_DEVICES
    before any CUDA call so it leaves no context on cuda:0. The 30-minute
    timeout covers per-rank cost imbalance during the final all_gather.
    """
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        physical_rank = int(os.environ.get("LOCAL_RANK", 0))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_rank)
        torch.cuda.set_device(0)
        if not torch.distributed.is_initialized():
            from datetime import timedelta
            torch.distributed.init_process_group(
                backend="nccl", timeout=timedelta(minutes=30))
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
        return rank, world, 0
    return 0, 1, 0


def load_model(model_path: str, adapter: Optional[str], device: torch.device,
               dino_id: str = DEFAULT_DINO_ID,
               dino_image_size: int = 256,
               dino_spatial_merge_size: int = 2,
               dino_deepstack_indexes=(6, 12, 18),
               prior_adapters: Optional[list] = None):
    """Load the dual-tower model, merge `prior_adapters` in order (Stage A),
    then merge `adapter` (Stage B). Returns (model, processor) in eval mode.

    Merging the prior adapters is essential for two-stage checkpoints: the
    Stage B adapter was trained on top of base+StageA, so skipping Stage A
    silently drops its contribution.
    """
    from src.model.model_patch import load_qwen3vl_dual
    print(f"[eval] loading dual-tower model (DINOv3 {dino_id}) from {model_path}", flush=True)
    model, processor = load_qwen3vl_dual(
        qwen_path=model_path,
        dino_id=dino_id,
        image_size=dino_image_size,
        spatial_merge_size=dino_spatial_merge_size,
        deepstack_indexes=tuple(dino_deepstack_indexes),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = model.to(device)
    from peft import PeftModel
    for pa in (prior_adapters or []):
        _mp = Path(pa) / "merger_weights.pt"
        if _mp.exists():
            _sd = torch.load(_mp, map_location="cpu")
            _, _unexp = model.load_state_dict(_sd, strict=False)
            print(f"[eval] loaded merger-only stage from {_mp} "
                  f"({len(_sd)} tensors, unexpected={len(_unexp)})", flush=True)
            continue
        print(f"[eval] merging prior adapter {pa}", flush=True)
        model = PeftModel.from_pretrained(model, pa)
        model = model.merge_and_unload()
    if adapter:
        print(f"[eval] applying LoRA adapter {adapter}", flush=True)
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return model, processor


def add_dual_model_args(p, default_model: str):
    """Register the model/adapter/DINO flags every evaluator shares."""
    p.add_argument("--model", default=default_model)
    p.add_argument("--adapter", default=None, help="Stage B adapter dir.")
    p.add_argument("--prior-adapter", action="append", default=None,
                   help="Earlier-stage adapter(s) merged BEFORE --adapter (Stage A).")
    p.add_argument("--dino-model-id", default=DEFAULT_DINO_ID)
    p.add_argument("--dino-image-size", type=int, default=256)
    p.add_argument("--dino-deepstack-indexes", default="6,12,18",
                   help="Comma-separated DINO tap layers; must match training.")
    p.add_argument("--dino-global-norm", action="store_true",
                   help="Delay-embedding global min/max normalisation (only if trained that way).")
    p.add_argument("--channel-stats-path", default=None,
                   help="JSON of {field -> {vmin, vmax}} for --dino-global-norm.")
    p.add_argument("--chart-style", default="v2",
                   help="Chart renderer; must match the training config (v2 default).")
    return p


def add_determinism_arg(p, default: bool):
    """`--deterministic / --no-deterministic`: bit-exact kernels (see
    `enable_determinism`). Each evaluator passes its own default."""
    p.add_argument("--deterministic", dest="deterministic", action="store_true", default=default,
                   help="Bit-exact kernels: pinned cuBLAS workspace, deterministic algorithms, "
                        "math-only SDPA (~2x slower attention)."
                        + (" Default on." if default else " Default off."))
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                   help="Fast, non-bit-exact kernels.")
    return p


def load_model_from_args(args, device):
    return load_model(
        args.model, args.adapter, device,
        dino_id=args.dino_model_id,
        dino_image_size=args.dino_image_size,
        dino_deepstack_indexes=tuple(int(x) for x in args.dino_deepstack_indexes.split(",")),
        prior_adapters=args.prior_adapter,
    )


def load_channel_stats(args, tag: str = "eval"):
    """Resolve the optional delay global-norm stats from CLI args
    (`--dino-global-norm` + `--channel-stats-path`); None = per-series norm."""
    if not getattr(args, "dino_global_norm", False):
        return None
    if not getattr(args, "channel_stats_path", None):
        raise SystemExit("--dino-global-norm needs --channel-stats-path (a {field -> {vmin, vmax}} JSON)")
    stats = json.load(open(args.channel_stats_path))
    print(f"[{tag}] global_norm from {args.channel_stats_path}", flush=True)
    return stats


def build_dual_collator(args, processor, channel_stats=None, max_length=None):
    """UnifiedDualCollator from the shared eval CLI flags."""
    from src.datasets_utils.unified_collator import UnifiedDualCollator
    return UnifiedDualCollator(
        processor=processor,
        max_length=max_length or getattr(args, "max_length", 4096),
        delay_image_size=args.dino_image_size,
        channel_stats=channel_stats,
        chart_style=getattr(args, "chart_style", None) or "v2",
    )


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------

def greedy_cfg(max_new_tokens: int):
    """Explicit greedy GenerationConfig: every sampling field is pinned so the
    model's own generation_config.json (do_sample=True, temperature 0.7, ...)
    cannot leak through."""
    from transformers import GenerationConfig
    return GenerationConfig(
        do_sample=False, max_new_tokens=max_new_tokens,
        temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
    )


def run_one(model, processor, example, collator, max_new_tokens: int = 16) -> str:
    """Greedy-generate one answer for a unified example through `collator`."""
    inputs = collator([example], for_generation=True)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=greedy_cfg(max_new_tokens))
    gen = out[:, inputs["input_ids"].shape[1]:]
    return strip_think(processor.tokenizer.batch_decode(gen, skip_special_tokens=False)[0])


def enable_determinism():
    """Bit-exact eval mode: pinned cuBLAS workspace + deterministic algorithms +
    math-only SDPA (~2x slower attention). Must run before the first CUDA call."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print("[determinism] bit-exact mode ON (pinned cuBLAS, math-SDPA)", flush=True)


# --------------------------------------------------------------------------
# Result provenance
# --------------------------------------------------------------------------

def dino_weights_sha256(dino_id: Optional[str]) -> Optional[str]:
    """SHA-256 of the DINOv3 `model.safetensors` that `dino_id` resolves to: for a
    Hub id the HF-cache blob name (which is the sha256 of an LFS file), for a
    local directory the hash of the file. None when it cannot be resolved."""
    if not dino_id:
        return None
    try:
        local = Path(dino_id) / "model.safetensors"
        if local.is_file():
            import hashlib
            h = hashlib.sha256()
            with open(local, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            return h.hexdigest()
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(dino_id, "model.safetensors")
        if isinstance(p, str) and os.path.islink(p):
            blob = os.path.basename(os.readlink(p))
            if re.fullmatch(r"[0-9a-f]{64}", blob):
                return blob
    except Exception:  # noqa: BLE001
        pass
    return None


def provenance(args, **extra) -> dict:
    """Fields every result JSON carries so a number can be traced back to the
    exact setup: checkpoints, chart renderer, token budgets, decoding, the
    full parsed command line (`cli_args`), and the code revision."""
    import datetime
    import subprocess
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=str(_REPO_ROOT), timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        rev = None
    d = {
        "model": getattr(args, "model", None),
        "prior_adapter": getattr(args, "prior_adapter", None),
        "adapter": getattr(args, "adapter", None),
        "chart_style": getattr(args, "chart_style", None) or "v2",
        "max_length": getattr(args, "max_length", None),
        "max_new_tokens": getattr(args, "max_new_tokens", None),
        "dino_model_id": getattr(args, "dino_model_id", None),
        "dino_weights_sha256": dino_weights_sha256(getattr(args, "dino_model_id", None)),
        "dino_image_size": getattr(args, "dino_image_size", None),
        "dino_deepstack_indexes": getattr(args, "dino_deepstack_indexes", None),
        "decoding": "greedy",
        "deterministic_kernels": getattr(args, "deterministic", None),
        "cli_args": {k: (v if v is None or isinstance(v, (bool, int, float)) else str(v))
                     for k, v in sorted(vars(args).items())},
        "git_commit": rev,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    d.update(extra)
    return d


# --------------------------------------------------------------------------
# Answer parsing and aggregation
# --------------------------------------------------------------------------

def extract_letter(text: str, n_options: int) -> Optional[str]:
    """Robustly extract A/B/C/... from a model's generated answer.

    Priority order is most specific -> least specific. Returns None on miss
    so the caller can log the failure.
    """
    if text is None:
        return None
    valid = set(LETTERS[:n_options])
    s = text.strip()

    patterns = [
        r"^\s*([A-H])\b",                       # leading letter
        r"answer\s*(?:is|:)?\s*\(?\s*([A-H])",  # "answer is A" / "Answer: A"
        r"\b([A-H])\s*\)",                      # "A)"
        r'"answer"\s*:\s*"?([A-H])',            # JSON-style {"answer":"A"}
        r"\b([A-H])\b",                         # any standalone letter
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            if letter in valid:
                return letter
    return None


def extract_letter_from_json(text: str, n_options: int) -> Optional[str]:
    """Parse a JSON object {reasoning, answer} (TSRBench style); fall back to
    plain regex extraction if parsing fails."""
    if text is None:
        return None
    valid = set(LETTERS[:n_options])
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            ans = obj.get("answer") if isinstance(obj, dict) else None
            if isinstance(ans, str):
                m2 = re.search(r"([A-H])", ans, flags=re.IGNORECASE)
                if m2 and m2.group(1).upper() in valid:
                    return m2.group(1).upper()
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    return extract_letter(text, n_options)


def per_group_accuracy(records: list, key: str) -> dict:
    """records: list of {"correct": bool, key: str}"""
    buckets = {}
    for r in records:
        k = r.get(key, "")
        b = buckets.setdefault(k, {"n": 0, "correct": 0})
        b["n"] += 1
        if r["correct"]:
            b["correct"] += 1
    return {k: {"n": v["n"], "correct": v["correct"],
                "acc": v["correct"] / v["n"] if v["n"] else 0.0}
            for k, v in sorted(buckets.items())}


def strip_think(text: str) -> str:
    """Drop a thinking block (up to and including '</think>') and chat special
    tokens from generated text. A no-op for non-thinking output."""
    if text is None:
        return text
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    for tok in ("<|im_end|>", "<|endoftext|>", "<think>"):
        text = text.replace(tok, "")
    return text.strip()
