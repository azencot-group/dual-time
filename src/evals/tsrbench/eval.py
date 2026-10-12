"""Evaluate the dual-tower model on TSRBench (12 multiple-choice tasks) in ONE run.

Every task is scored under the protocol the model can actually use, baked in
per task (README.md):

  * ten tasks inline: single-letter prompt, series-as-choices items (deductive
    reasoning) rendered as inline `A) <ts><ts/>` charts, 8,192-token budget
    (= training);
  * time_series_forecasting: the four candidate continuations as plain text
    (history still encoded), 16,384-token budget -- as four small charts the
    permuted candidates are indistinguishable, and the 140-step items need
    ~9k prompt tokens;
  * abductive_reasoning: the harness's game context (past/future events +
    critical-moment marker) prepended, 16,384-token budget.

Each series is rendered both as a chart (native Qwen ViT) and as a
delay-embedding image (DINOv3), exactly as in training. One greedy generate per
item; items are sharded across torchrun ranks and gathered on rank 0. Output:
one JSON (`tsrbench.json`) with `overall` (accuracy over the 4,125 items),
`per_task`, the records and the provenance block.

Example:
  torchrun --nproc_per_node=2 -m src.evals.tsrbench.eval \
      --model artifacts/models/Qwen3-VL-8B-Instruct \
      --prior-adapter artifacts/checkpoints/stageA_dual8b/best \
      --adapter artifacts/checkpoints/stageB_dual8b/checkpoint-5714 \
      --max-new-tokens 1024 --out results/<label>/tsrbench.json
`python scripts/eval.py --kind tsrb` drives it; --tasks filters to a subset
for quick checks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src import paths
from src.evals.common import (DEFAULT_MODEL, add_determinism_arg, add_dual_model_args,
                              build_dual_collator, enable_determinism, extract_letter,
                              extract_letter_from_json, init_distributed, load_model_from_args,
                              per_group_accuracy, provenance, run_one)
from .data import ALL_TASKS, load_tsrbench_unified

DEFAULT_DATASET = paths.resolve("eval_data/tsrbench")
INLINE_MAX_LENGTH = 8192    # training budget; every inline task fits
TEXT_MAX_LENGTH = 16384     # plain-text forecasting / abductive with context need 6-9k prompt tokens


def task_protocol(task: str, inline_max_length: int = INLINE_MAX_LENGTH,
                  text_max_length: int = TEXT_MAX_LENGTH) -> dict:
    """The reported protocol of one task: how series-as-choices items are
    rendered, whether the abductive game context is prepended, and the
    collator token budget."""
    if task == "time_series_forecasting":
        return {"series_choices_inline": False, "ar_context": False, "max_length": text_max_length,
                "note": "plain-text candidates, history encoded"}
    if task == "abductive_reasoning":
        return {"series_choices_inline": False, "ar_context": True, "max_length": text_max_length,
                "note": "game context prepended"}
    return {"series_choices_inline": True, "ar_context": False, "max_length": inline_max_length,
            "note": "inline series-as-choices"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dataset-dir", default=DEFAULT_DATASET)
    p.add_argument("--tasks", default="all",
                   help="comma-separated TSRBench tasks (quick checks); default = all 12.")
    p.add_argument("--prompt-style", choices=["qa", "tsrbench"], default="qa",
                   help="qa = single-letter (what the model was trained on, default); "
                        "tsrbench = official JSON-with-reasoning prompt.")
    p.add_argument("--limit-per-task", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--inline-max-length", type=int, default=INLINE_MAX_LENGTH,
                   help="Collator token budget of the ten inline tasks (training used 8192).")
    p.add_argument("--text-max-length", type=int, default=TEXT_MAX_LENGTH,
                   help="Collator token budget of plain-text forecasting and abductive reasoning "
                        "with context (8k silently truncates them).")
    p.add_argument("--ar-all-channels", action="store_true",
                   help="abductive_reasoning: show every channel instead of the win-probability ones.")
    p.add_argument("--ar-full-series", action="store_true",
                   help="abductive_reasoning: show history + future instead of the history only.")
    add_dual_model_args(p, DEFAULT_MODEL)
    # Deterministic by default for TSRBench (--no-deterministic opts out).
    add_determinism_arg(p, default=True)
    args = p.parse_args()
    if args.deterministic:
        enable_determinism()   # before the first CUDA call

    if args.tasks.strip().lower() == "all":
        tasks = list(ALL_TASKS)
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    protocols = {t: task_protocol(t, args.inline_max_length, args.text_max_length) for t in tasks}

    rank, world, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"[eval_tsrbench] rank={rank}/{world} device={device}", flush=True)

    examples = []
    for task in tasks:
        proto = protocols[task]
        rows = load_tsrbench_unified(
            args.dataset_dir, tasks=[task], prompt_style=args.prompt_style,
            limit_per_task=args.limit_per_task,
            series_choices_inline=proto["series_choices_inline"],
            ar_context=proto["ar_context"],
            ar_all_channels=args.ar_all_channels, ar_full_series=args.ar_full_series,
        )
        for e in rows:
            m = e["meta"]
            m["idx"] = len(examples)          # global idx across tasks, in ALL_TASKS order
            e.update(idx=m["idx"], task=task, group=m["group"],
                     gold_letter=m["gold_letter"], options=[None] * m["n_options"],
                     max_length=proto["max_length"])
            examples.append(e)
    my_examples = examples[rank::world]
    print(f"[eval_tsrbench] rank={rank} -> {len(my_examples)} items", flush=True)

    model, processor = load_model_from_args(args, device)
    channel_stats = None
    if args.channel_stats_path:
        # TSRBench field names rarely match training stats; unmatched series
        # fall back to per-series instance norm.
        with open(args.channel_stats_path) as f:
            channel_stats = json.load(f)
    collators = {ml: build_dual_collator(args, processor, channel_stats, max_length=ml)
                 for ml in sorted({pr["max_length"] for pr in protocols.values()})}

    extract_fn = extract_letter_from_json if args.prompt_style == "tsrbench" else extract_letter
    if args.prompt_style == "tsrbench" and args.max_new_tokens < 256:
        args.max_new_tokens = 512  # JSON-with-reasoning output needs room

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    miss_log_path = args.out.replace(".json", "_parse_misses.jsonl")

    records = []
    t0 = time.time()
    for ex in tqdm(my_examples, desc=f"rank{rank}", disable=(rank != 0)):
        try:
            raw = run_one(model, processor, ex, collators[ex["max_length"]], args.max_new_tokens)
        except Exception as e:
            raw = f"<<error:{type(e).__name__}: {e}>>"
        pred = extract_fn(raw, n_options=len(ex["options"]))
        rec = {
            "idx": ex["idx"], "task": ex["task"], "group": ex["group"],
            "gold": ex["gold_letter"], "pred": pred, "raw": raw,
            "correct": bool(pred == ex["gold_letter"]),
        }
        if pred is None:
            with open(miss_log_path + f".rank{rank}", "a") as f:
                f.write(json.dumps(rec) + "\n")
        records.append(rec)

    if world > 1:
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, records)
        if rank != 0:
            return
        records = [r for sub in gathered for r in sub]

    records.sort(key=lambda r: r["idx"])
    n = len(records)
    n_correct = sum(r["correct"] for r in records)
    n_miss = sum(r["pred"] is None for r in records)
    report = {
        "benchmark": "tsrbench",
        "model": args.model,
        "adapter": args.adapter,
        "prior_adapter": args.prior_adapter,
        "prompt_style": args.prompt_style,
        "tasks": tasks,
        "n": n,
        "correct": n_correct,
        "overall": n_correct / n if n else 0.0,
        "parse_misses": n_miss,
        "elapsed_seconds": time.time() - t0,
        "provenance": provenance(args, dataset_dir=args.dataset_dir, protocols=protocols,
                                 limit_per_task=args.limit_per_task, world_size=world),
        "per_task": per_group_accuracy(records, "task"),
        "per_group": per_group_accuracy(records, "group"),
        "records": records,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[eval_tsrbench] DONE overall={report['overall']:.4f} "
          f"({n_correct}/{n}) parse_misses={n_miss} -> {args.out}")
    for task, s in report["per_task"].items():
        print(f"  {task:30s} {s['correct']:4d}/{s['n']:<4d} acc={s['acc']:.3f}   [{protocols[task]['note']}]")
    print("  --- groups ---")
    for grp, s in report["per_group"].items():
        print(f"  {grp:30s} {s['correct']:4d}/{s['n']:<4d} acc={s['acc']:.3f}")


if __name__ == "__main__":
    main()
