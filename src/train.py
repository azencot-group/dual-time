"""Stage A / Stage B LoRA SFT for the dual-tower Qwen3-VL-8B time-series model.

The model is Qwen3-VL-8B-Instruct with two vision towers (see model_patch):
  * the native Qwen ViT reads a matplotlib chart of every series (image stream);
  * a DINOv3 ViT-L/16 reads a delay-embedding image of the same series
    (video stream).

Training data is the unified example schema from chatts_data.py (`<ts><ts/>`
markers + free-text targets); UnifiedDualCollator renders the charts and delay
images on the fly. Launched with accelerate + DeepSpeed (python scripts/train.py).

Two stages, selected purely by config keys:
  Stage A  dino_lora_on_backbone: true, dino_lora_on_lm: false
           -> LoRA on the DINOv3 backbone + full-weight DINO mergers, LM frozen.
  Stage B  resume_adapter: <stage A>/best, resume_mode: merge,
           dino_lora_on_backbone: false, dino_lora_on_lm: true,
           dino_freeze_merger: true
           -> Stage A merged into the base, fresh LoRA on the LM only.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from . import paths

# torch / numpy / trl are imported inside the functions that need them so that
# `python -m src.train --help` (build_parser) works without the training stack.

DEFAULT_DINO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def set_seed(seed: int):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _list_dataset(items):
    """Wrap a list of examples as a torch Dataset."""
    from torch.utils.data import Dataset

    class _ListDS(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    return _ListDS(items)


def setup_lora(model, cfg):
    """Attach a fresh LoRA adapter according to the stage knobs in `cfg`.

    Knobs (all optional):
      freeze_vision           freeze both vision towers before LoRA (default true)
      dino_freeze_merger      keep the DINO mergers frozen (Stage B); otherwise they
                              train full-weight via PEFT `modules_to_save`
      dino_lora_on_backbone   LoRA on the DINOv3 ViT (Stage A)
      dino_lora_on_lm         LoRA on the language model (Stage B)
    The native Qwen ViT (`visual`) is never LoRA'd. With both LoRA knobs off
    the mergers are trained alone and saved as best/merger_weights.pt.
    """
    from peft import LoraConfig, get_peft_model

    if cfg.get("freeze_vision", True):
        for name, p in model.named_parameters():
            if "visual" in name or "vision" in name:
                p.requires_grad = False

    tower = model.model.visual_dino
    n_deepstack = len(tower.deepstack_merger_list)
    modules_to_save = None
    if not bool(cfg.get("dino_freeze_merger", False)):
        merger_substrings = ("visual_dino.merger", "visual_dino.deepstack_merger_list")
        for name, p in model.named_parameters():
            if any(s in name for s in merger_substrings):
                p.requires_grad = True
        # PEFT refuses to wrap a ModuleList, so list the indexed children.
        modules_to_save = ["visual_dino.merger"] + [
            f"visual_dino.deepstack_merger_list.{i}" for i in range(n_deepstack)]
    else:
        print("[setup_lora] dino_freeze_merger=True: DINO mergers stay frozen", flush=True)

    lora_backbone = bool(cfg.get("dino_lora_on_backbone", True))
    lora_lm = bool(cfg.get("dino_lora_on_lm", True))
    if not lora_backbone and not lora_lm:
        if bool(cfg.get("dino_freeze_merger", False)):
            raise ValueError("no-LoRA mode needs a trainable merger: set dino_freeze_merger=false")
        n_train = 0
        for name, p in model.named_parameters():
            train_it = ("visual_dino" in name and "merger" in name)
            p.requires_grad = train_it
            if train_it:
                n_train += p.numel()
        print(f"[setup_lora] no-LoRA mode: merger-only training "
              f"({n_train/1e6:.1f}M params, no PEFT adapter)", flush=True)
        return model

    # LoRA target names (q_proj, ...) collide between the LM and the DINOv3
    # backbone, so scope the adapter with exclude patterns.
    excludes = [r".*\.visual\..*"]
    if not lora_backbone:
        excludes.append(r".*visual_dino\.backbone\..*")
    if not lora_lm:
        excludes.append(r".*language_model\..*")
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        target_modules=cfg["lora_target_modules"],
        modules_to_save=modules_to_save,
        exclude_modules="(" + "|".join(excludes) + ")",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _setup_continue(model, cfg):
    """An existing PEFT adapter is attached and trainable (is_trainable=True).
    No fresh LoRA is created; keep training the same matrices. Freezes the
    mergers when dino_freeze_merger=true and re-enables input grads for
    gradient checkpointing."""
    if bool(cfg.get("dino_freeze_merger", False)):
        n = 0
        for name, p in model.named_parameters():
            if "merger" in name:
                p.requires_grad = False
                n += 1
        print(f"[setup_continue] dino_freeze_merger=True: froze {n} merger tensors", flush=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[setup_continue] continuing existing adapter: trainable "
          f"{n_train/1e6:.2f}M / total {n_total/1e6:.0f}M", flush=True)
    return model


def _slim_deepspeed_checkpoints():
    """Mergers-only training (no PEFT adapter): HF lets DeepSpeed save the WHOLE
    module (~17 GB) in every checkpoint and reloads it strictly. Save only the
    trainable tensors and load non-strictly instead -- exactly what HF does for
    PEFT models -- so a checkpoint is a few hundred MB."""
    try:
        from deepspeed.runtime.engine import DeepSpeedEngine
    except ImportError:  # plain accelerate run: nothing to slim
        return
    if getattr(DeepSpeedEngine, "_dual_time_slim", False):
        return
    _save, _load = DeepSpeedEngine.save_checkpoint, DeepSpeedEngine.load_checkpoint

    def save_checkpoint(self, *a, **kw):
        kw["exclude_frozen_parameters"] = True
        return _save(self, *a, **kw)

    def load_checkpoint(self, *a, **kw):
        kw["load_module_strict"] = False
        return _load(self, *a, **kw)

    DeepSpeedEngine.save_checkpoint, DeepSpeedEngine.load_checkpoint = save_checkpoint, load_checkpoint
    DeepSpeedEngine._dual_time_slim = True
    print("[train] mergers-only mode: DeepSpeed checkpoints carry trainable tensors only", flush=True)


def _run_trainer(cfg, args, model, processor, train_ds, collator,
                 sample_weights, BalancedSFTTrainer, SFTConfig):
    """Build the SFT trainer, run training, then write best/ from the last checkpoint."""
    import torch
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    from peft import PeftModel as _PeftModel
    # A Stage A without a backbone LoRA (mergers only): no PEFT
    # adapter, so checkpoints hold just the trainable merger tensors.
    merger_keys = None
    if not isinstance(model, _PeftModel):
        merger_keys = sorted(n for n, p in model.named_parameters() if p.requires_grad)
        _slim_deepspeed_checkpoints()
    # Rolling checkpoints (default on): RollingCheckpointCallback owns rotation,
    # so HF's save_total_limit must be off. See preemption.py.
    rolling = bool(cfg.get("rolling_checkpoints", True))
    save_total_limit = None if rolling else cfg.get("save_total_limit")
    sft_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs=(
            {"use_reentrant": bool(cfg.get("gc_use_reentrant", False))}
            if cfg["gradient_checkpointing"] else None),
        logging_steps=cfg["logging_steps"],
        # Resumable checkpoints: save_only_model=False keeps optimizer,
        # scheduler and RNG state so --resume continues exactly.
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=int(cfg.get("save_steps", 500)),
        save_total_limit=save_total_limit,
        seed=cfg.get("seed", 42),
        report_to=["tensorboard"],
        logging_dir=str(out_dir / "tb"),
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_steps=args.max_steps if args.max_steps else -1,
        save_only_model=bool(cfg.get("save_only_model", False)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 0)),
    )

    from .preemption import PreemptionCallback, RollingCheckpointCallback
    callbacks = [PreemptionCallback()]
    if rolling:
        callbacks.append(RollingCheckpointCallback(
            output_dir=str(out_dir),
            keep_full=int(cfg.get("keep_full_checkpoints", 2)),
            keep_epoch_adapters=cfg.get("keep_epoch_adapters", True),
        ))
    preempt_cb = callbacks[0]
    trainer = BalancedSFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=processor.tokenizer,
        sample_weights=sample_weights,
        merger_keys=merger_keys,
        callbacks=callbacks,
    )

    # --resume: continue from the newest checkpoint in output_dir if there is
    # one, otherwise start fresh.
    resume_ckpt = args.resume
    if args.resume:
        # Pick the newest checkpoint that is actually COMPLETE. A preemption whose
        # grace period expires mid-save leaves a truncated directory (no
        # trainer_state.json); Trainer would then abort with FileNotFoundError and
        # the whole run is lost even though an older, intact checkpoint exists.
        def _complete(d):
            return (d / "trainer_state.json").is_file() and (
                (d / "latest").is_file() or (d / "optimizer.pt").is_file()
                or any(d.glob("global_step*")))
        last = None
        if out_dir.exists():
            for cand in sorted(out_dir.glob("checkpoint-*"),
                               key=lambda q: int(q.name.split("-")[-1]), reverse=True):
                if _complete(cand):
                    last = str(cand); break
                print(f"[train] --resume: skipping incomplete checkpoint {cand.name} "
                      f"(no trainer_state.json / optimizer state)", flush=True)
        resume_ckpt = last if last else False
        print(f"[train] --resume: {'resuming from ' + last if last else 'no checkpoint yet, starting fresh'}",
              flush=True)

    print("[train] starting", flush=True)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    if preempt_cb.preempted:
        # A full checkpoint was just written; the relaunch resumes from it and
        # finalizes best/ when training actually completes.
        print("[train] stopped for preemption; checkpoint saved.", flush=True)
        return

    if trainer.is_world_process_zero():
        _inner = getattr(trainer.model, "module", trainer.model)
        ckpts = sorted(out_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        if not isinstance(_inner, _PeftModel):
            # Mergers-only mode: best/ = the trained tensors of the newest
            # checkpoint (merger_weights.pt written by BalancedSFTTrainer._save).
            best_dir = out_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            merger_sd = {}
            if ckpts and (ckpts[-1] / "merger_weights.pt").exists():
                merger_sd = torch.load(ckpts[-1] / "merger_weights.pt", map_location="cpu")
            elif ckpts and (ckpts[-1] / "model.safetensors.index.json").exists():
                import json as _json
                from safetensors.torch import load_file as _lf
                _idx = _json.load(open(ckpts[-1] / "model.safetensors.index.json"))
                _shards = {}
                for k, sh in _idx["weight_map"].items():
                    if "visual" in k and "merger" in k:
                        if sh not in _shards:
                            _shards[sh] = _lf(str(ckpts[-1] / sh))
                        merger_sd[k] = _shards[sh][k].detach().cpu()
            if not merger_sd:
                merger_sd = {k: p.detach().cpu() for k, p in _inner.named_parameters()
                             if p.requires_grad}
            torch.save(merger_sd, best_dir / "merger_weights.pt")
            print(f"[train] saved merger-only weights ({len(merger_sd)} tensors) to "
                  f"{best_dir}/merger_weights.pt", flush=True)
            return
        if ckpts:
            best_dir = out_dir / "best"
            import shutil
            if best_dir.exists():
                shutil.rmtree(best_dir)
            # best/ = adapter + tokenizer only; optimizer/scheduler/RNG state is
            # dropped so it stays small and loads cleanly as a prior adapter.
            shutil.copytree(ckpts[-1], best_dir, ignore=shutil.ignore_patterns(
                "optimizer*", "scheduler*", "rng_state*", "global_step*",
                "latest", "*.distcp", "trainer_state.json"))
            print(f"[train] saved adapter to {best_dir} (from {ckpts[-1].name})", flush=True)
        else:
            print("[train] WARNING: no checkpoint-* dirs found; nothing copied to best/")


def load_model_from_cfg(cfg):
    """Build the dual-tower model + processor described by a training config."""
    import torch
    from src.model.model_patch import load_qwen3vl_dual
    print(f"[train] loading dual-tower model from {cfg['model_path']}", flush=True)
    return load_qwen3vl_dual(
        qwen_path=cfg["model_path"],
        dino_id=cfg.get("dino_model_id", DEFAULT_DINO_ID),
        image_size=int(cfg.get("dino_image_size", 256)),
        spatial_merge_size=int(cfg.get("dino_spatial_merge_size", 2)),
        deepstack_indexes=tuple(cfg.get("dino_deepstack_indexes", (6, 12, 18))),
        freeze_dino_backbone=bool(cfg.get("dino_freeze_backbone", True)),
        dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )


def build_sample_weights(cfg, train_examples):
    """Per-example sampling weights for the WeightedRandomSampler, or None.

    balance_mode:
      weighted_even      every distinct `task` gets an equal expected share
                         (1/K) per epoch via 1/count(task) weighting.
      weighted_even_src  like weighted_even, but ChatTS subsets (meta['subset'])
                         become separate buckets so ift is not drowned by sft.
                         `bucket_repetition: {bucket: R}` pins the expected
                         per-row repetition of a bucket to exactly R per epoch;
                         the remaining budget is split evenly across the rest.
    balance_by_meta: <key>  weight by 1/count(meta[key]) instead.
    """
    import collections
    balance_mode = cfg.get("balance_mode")
    balance_meta = cfg.get("balance_by_meta")
    N = len(train_examples)
    if balance_mode == "weighted_even":
        counts = collections.Counter(ex.get("task") for ex in train_examples)
        K = len(counts)
        print(f"[train] balance_mode=weighted_even: N={N}, K={K} tasks, "
              f"counts={dict(counts)}", flush=True)
        return [(N / (K * counts[ex.get("task")])) if counts.get(ex.get("task")) else 0.0
                for ex in train_examples]
    if balance_mode == "weighted_even_src":
        def _src_bucket(ex):
            t = ex.get("task")
            sub = (ex.get("meta") or {}).get("subset")
            return f"{t}:{sub}" if (t == "chatts" and sub) else t
        keys = [_src_bucket(ex) for ex in train_examples]
        counts = collections.Counter(keys)
        K = len(counts)
        pin = dict(cfg.get("bucket_repetition") or {})
        if pin:
            pinned_draws = sum(pin[b] * counts[b] for b in pin if b in counts)
            others = [b for b in counts if b not in pin]
            rem_share = max(0.0, N - pinned_draws) / max(1, len(others))
            w_by_bucket = {b: (pin[b] if b in pin else rem_share / counts[b]) for b in counts}
            reps = {b: round(w_by_bucket[b] if b in pin else rem_share / counts[b], 2)
                    for b in counts}
            print(f"[train] balance_mode=weighted_even_src+pin: N={N}, "
                  f"buckets={dict(counts)}, repetition/epoch={reps}", flush=True)
            return [w_by_bucket[k] for k in keys]
        print(f"[train] balance_mode=weighted_even_src: N={N}, K={K} buckets, "
              f"counts={dict(counts)}", flush=True)
        return [(N / (K * counts[k])) if counts.get(k) else 0.0 for k in keys]
    if balance_meta:
        counts = collections.Counter(ex.get("meta", {}).get(balance_meta) for ex in train_examples)
        K = len(counts)
        print(f"[train] balance_by_meta='{balance_meta}': {K} groups, {N} examples "
              f"(min/max group size {min(counts.values())}/{max(counts.values())})", flush=True)
        return [N / (K * counts[ex.get("meta", {}).get(balance_meta)]) for ex in train_examples]
    if balance_mode:
        raise ValueError(f"unknown balance_mode: {balance_mode!r}")
    return None


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return paths.resolve_cfg_paths(cfg)  # repo-relative paths -> absolute (src/paths.py)


# Explicit "where things are" flags: (flag, config key, type, help). Every other
# config key (data recipe, towers, LoRA, optimisation) stays in the YAML and is
# overridden with --set KEY=VALUE.
_LOCATION_FLAGS = (
    ("--output-dir", "output_dir", str,
     "Checkpoint directory (checkpoint-*/, best/, tb/)."),
    ("--model-path", "model_path", str,
     "Base Qwen3-VL-8B-Instruct: a local directory or a Hub id."),
    ("--resume-adapter", "resume_adapter", str,
     "Stage A prior merged into the base before the LM LoRA is attached "
     "(Stage B; a PEFT adapter dir such as artifacts/checkpoints/stageA_dual8b/best)."),
    ("--dino-model-id", "dino_model_id", str,
     "DINOv3 tower: a Hub id or a local directory."),
    ("--save-steps", "save_steps", int,
     "Write a resumable checkpoint every N optimizer steps."),
    ("--seed", "seed", int,
     "Random seed (data shuffle, sampler, model init)."),
)

_EPILOG = """\
Everything else -- the data recipe (unified_sources, balance_mode,
bucket_repetition), the dual-tower settings (dino_*, attn_implementation,
max_length, chart_style), the LoRA shape (lora_r, lora_alpha, lora_dropout,
lora_target_modules) and the optimisation (num_train_epochs, learning_rate,
lr_scheduler_type, warmup_ratio, weight_decay, per_device_train_batch_size,
gradient_accumulation_steps, gradient_checkpointing, ...) -- is set in the
config; --set KEY=VALUE overrides any key (the value is YAML-parsed, so lists
and numbers work: --set lora_r=64 --set 'dino_deepstack_indexes=[6,12,18]').

Precedence: config file < explicit flags above < --set (applied in order).
Path-valued keys (output_dir, *_path, *_adapter) given relative to the
repository root are resolved the same way as in the config file.

examples:
  python -m src.train --config configs/stageA_dual8b.yaml
  python -m src.train --config configs/stageB_dual8b.yaml \\
      --output-dir artifacts/checkpoints/myrun \\
      --resume-adapter artifacts/checkpoints/stageA_dual8b/best
  python -m src.train --config configs/stageB_dual8b.yaml --max-steps 20 --set lora_r=64

Normally launched through scripts/train.py (accelerate + DeepSpeed), which
forwards any flag it does not know to this module.
"""


def build_parser() -> argparse.ArgumentParser:
    """The src.train command line (argparse only; no torch needed for --help)."""
    p = argparse.ArgumentParser(
        prog="python -m src.train",
        description="Stage A / Stage B LoRA SFT of the dual-tower Qwen3-VL-8B "
                    "time-series model (one training process; launch with accelerate).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG)
    cfgg = p.add_argument_group("config")
    cfgg.add_argument("--config", required=True, metavar="PATH",
                      help="Training config YAML (configs/stageA_dual8b.yaml, configs/stageB_dual8b.yaml).")
    cfgg.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                      help="Override any config value (YAML-parsed); repeatable, e.g. "
                           "--set per_device_train_batch_size=4 --set attn_implementation=sdpa")
    loc = p.add_argument_group("locations (default: the config value)")
    for flag, key, typ, help_ in _LOCATION_FLAGS:
        loc.add_argument(flag, dest=key, type=typ, default=None,
                         metavar="PATH" if typ is str else "N",
                         help=f"{help_} [config key: {key}]")
    run = p.add_argument_group("run control")
    run.add_argument("--max-steps", type=int, default=None, metavar="N",
                     help="Stop after N optimizer steps (quick checks).")
    run.add_argument("--resume", action="store_true",
                     help="Resume from the newest intact checkpoint in output_dir if one exists.")
    return p


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Explicit location flags first, then --set KEY=VALUE, each printed."""
    def _override(key, val, source):
        val = paths.resolve_cfg_paths(val, key)   # same path rule as the config file
        print(f"[config] override {key}: {cfg.get(key)!r} -> {val!r}  ({source})", flush=True)
        cfg[key] = val

    for flag, key, _typ, _help in _LOCATION_FLAGS:
        val = getattr(args, key)
        if val is not None:
            _override(key, val, flag)
    for kv in args.set:
        key, sep, raw = kv.partition("=")
        if not key or not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {kv!r}")
        _override(key, yaml.safe_load(raw), "--set")
    return cfg


def main():
    args = build_parser().parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    set_seed(cfg.get("seed", 42))

    import torch
    from trl import SFTTrainer, SFTConfig
    from torch.utils.data import WeightedRandomSampler

    class BalancedSFTTrainer(SFTTrainer):
        """SFTTrainer with an optional weighted training sampler
        (weights from build_sample_weights)."""
        def __init__(self, *args, sample_weights=None, merger_keys=None, **kwargs):
            self._sample_weights = sample_weights
            self._merger_keys = merger_keys   # non-None = mergers-only mode
            super().__init__(*args, **kwargs)

        def _save(self, output_dir=None, state_dict=None):
            # Mergers-only mode: write just the trainable tensors (+ tokenizer)
            # instead of the full model; see _slim_deepspeed_checkpoints.
            if self._merger_keys is None:
                return super()._save(output_dir, state_dict)
            output_dir = output_dir or self.args.output_dir
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            if state_dict is None:
                state_dict = self.model.state_dict()
            keep = {k: v.detach().cpu() for k, v in state_dict.items() if k in set(self._merger_keys)}
            torch.save(keep, Path(output_dir) / "merger_weights.pt")
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
            torch.save(self.args, Path(output_dir) / "training_args.bin")
            print(f"[train] saved {len(keep)} trainable tensors -> {output_dir}/merger_weights.pt", flush=True)

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            # Non-DeepSpeed resume in mergers-only mode (DeepSpeed resumes from
            # its own engine checkpoint and never reaches this).
            mp = Path(resume_from_checkpoint) / "merger_weights.pt"
            if self._merger_keys is not None and mp.exists():
                sd = torch.load(mp, map_location="cpu")
                (model or self.model).load_state_dict(sd, strict=False)
                print(f"[train] loaded {len(sd)} trainable tensors from {mp}", flush=True)
                return
            return super()._load_from_checkpoint(resume_from_checkpoint, model)

        def _get_train_sampler(self, train_dataset=None):
            if self._sample_weights is None:
                return super()._get_train_sampler(train_dataset)
            ds = train_dataset if train_dataset is not None else self.train_dataset
            return WeightedRandomSampler(
                weights=torch.as_tensor(self._sample_weights, dtype=torch.double),
                num_samples=len(ds),
                replacement=True,
            )

    model, processor = load_model_from_cfg(cfg)
    if cfg.get("gradient_checkpointing"):
        model.config.use_cache = False
        # gc_use_reentrant: true switches to reentrant checkpointing (needed by
        # some ZeRO-3 + mostly-frozen-model combinations).
        _reent = bool(cfg.get("gc_use_reentrant", False))
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": _reent})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # resume_adapter: bake earlier-stage adapter(s) into the base before the
    # fresh LoRA is attached (Stage B on top of Stage A). A list is merged in
    # order. resume_mode=continue keeps the LAST adapter trainable instead.
    resume_adapter = cfg.get("resume_adapter")
    resume_mode = cfg.get("resume_mode", "merge")
    resumed_peft = False
    if resume_adapter:
        from peft import PeftModel
        adapters = list(resume_adapter) if isinstance(resume_adapter, (list, tuple)) else [resume_adapter]
        if resume_mode == "continue":
            for a in adapters[:-1]:
                print(f"[train] merging adapter from {a}", flush=True)
                model = PeftModel.from_pretrained(model, a)
                model = model.merge_and_unload()
            last = adapters[-1]
            print(f"[train] continue-mode: loading PEFT adapter (trainable) from {last}", flush=True)
            model = PeftModel.from_pretrained(model, last, is_trainable=True)
            resumed_peft = True
        else:
            for a in adapters:
                _mp = Path(a) / "merger_weights.pt"
                if _mp.exists():
                    _sd = torch.load(_mp, map_location="cpu")
                    _, _unexp = model.load_state_dict(_sd, strict=False)
                    print(f"[train] loaded merger-only stage from {_mp} "
                          f"({len(_sd)} tensors, unexpected={len(_unexp)})", flush=True)
                    continue
                print(f"[train] merging adapter from {a}", flush=True)
                model = PeftModel.from_pretrained(model, a)
                model = model.merge_and_unload()
            print(f"[train] {len(adapters)} adapter(s) merged into base", flush=True)

    if resumed_peft:
        model = _setup_continue(model, cfg)
    elif cfg.get("continue_adapter"):
        # Warm start: attach an EXISTING adapter as trainable with a fresh
        # optimizer and LR schedule (unlike --resume, which restores both).
        from peft import PeftModel
        ca = cfg["continue_adapter"]
        print(f"[train] continue_adapter: attaching trainable LoRA from {ca}", flush=True)
        model = PeftModel.from_pretrained(model, ca, is_trainable=True)
        model = _setup_continue(model, cfg)
    else:
        model = setup_lora(model, cfg)

    unified_sources = cfg.get("unified_sources")
    if not unified_sources:
        raise ValueError("config needs `unified_sources` (see README.md)")
    from src.datasets_utils.sources import load_unified_sources
    from src.datasets_utils.unified_collator import UnifiedDualCollator

    train_examples = load_unified_sources(unified_sources)
    random.Random(cfg.get("seed", 42)).shuffle(train_examples)
    print(f"[train] unified train={len(train_examples)}", flush=True)
    train_ds = _list_dataset(train_examples)

    channel_stats = None
    if cfg.get("dino_global_norm", False):
        csp = cfg.get("channel_stats_path")
        if csp:
            import json as _json
            with open(csp) as _f:
                channel_stats = _json.load(_f)
            print(f"[train] dino_global_norm from {csp}: channel_stats={channel_stats}", flush=True)
        else:
            raise ValueError("dino_global_norm: true needs channel_stats_path (a {field -> {vmin, vmax}} JSON)")
    collator = UnifiedDualCollator(
        processor=processor,
        max_length=cfg.get("max_length", 4096),
        delay_image_size=int(cfg.get("dino_image_size", 256)),
        delay_patch_size=int(cfg.get("dino_patch_size", 16)),
        delay_spatial_merge_size=int(cfg.get("dino_spatial_merge_size", 2)),
        channel_stats=channel_stats,
        chart_style=str(cfg.get("chart_style", "v2")),
        max_chart_tokens_v2=int(cfg.get("max_chart_tokens_v2", 240)),
    )
    sample_weights = build_sample_weights(cfg, train_examples)
    _run_trainer(cfg, args, model, processor, train_ds,
                 collator, sample_weights, BalancedSFTTrainer, SFTConfig)


if __name__ == "__main__":
    main()
