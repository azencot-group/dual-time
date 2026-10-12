"""TimeSeriesExam (multiple choice) evaluation for the dual-tower model.

Items are sharded across ranks when launched with torchrun and gathered on
rank 0. The model loader and the distributed init live in src.evals.common.

Usage:
  torchrun --nproc_per_node=2 -m src.evals.tsexam.eval \
      --model artifacts/models/Qwen3-VL-8B-Instruct \
      --prior-adapter artifacts/checkpoints/stageA_dual8b/best \
      --adapter artifacts/checkpoints/stageB_dual8b/checkpoint-5714 \
      --max-new-tokens 1024 --out results/tsexam.json
`python scripts/eval.py --kind tsexam` drives it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.evals.common import (DEFAULT_DINO_ID, DEFAULT_MODEL, add_determinism_arg,
                              enable_determinism, extract_letter, extract_letter_from_json,
                              init_distributed, load_channel_stats, load_model, per_group_accuracy,
                              provenance, run_one)
from .data import BENCHMARK_REVISION, benchmark_rows_to_examples, load_raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None, help="Stage B adapter dir.")
    p.add_argument("--prior-adapter", action="append", default=None,
                   help="Earlier-stage adapter merged BEFORE --adapter (Stage A). Repeat to chain.")
    p.add_argument("--out", required=True)
    p.add_argument("--data-json", default=None,
                   help="Local TimeSeriesExam-schema JSON instead of the HF AutonLab/TimeSeriesExam1 set.")
    p.add_argument("--limit", type=int, default=None, help="Cap number of items (quick check).")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--max-length", type=int, default=6144)
    p.add_argument("--prompt-style", choices=["qa", "tsrbench"], default="qa",
                   help="qa = letter-only system+user (default); "
                        "tsrbench = JSON-with-reasoning prompt, no system role.")
    p.add_argument("--chart-style", default="v2",
                   help="Chart renderer; must match the training config (v2 default).")
    p.add_argument("--dino-model-id", default=DEFAULT_DINO_ID,
                   help="Must match the DINO backbone the adapters were trained against.")
    p.add_argument("--dino-image-size", type=int, default=256)
    p.add_argument("--dino-deepstack-indexes", default="6,12,18",
                   help="Comma-separated DINO tap layers; must match training.")
    p.add_argument("--dino-global-norm", action="store_true",
                   help="Delay-embedding global min/max normalisation (only if trained that way).")
    p.add_argument("--channel-stats-path", default=None,
                   help="JSON of {field -> {vmin, vmax}} for --dino-global-norm.")
    add_determinism_arg(p, default=False)
    args = p.parse_args()
    if args.deterministic:
        enable_determinism()   # before the first CUDA call

    rank, world, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"[eval] rank={rank}/{world} device={device}", flush=True)

    from src.datasets_utils.unified_collator import UnifiedDualCollator
    rows = load_raw(json_path=args.data_json)
    indices = list(range(len(rows)))
    if args.limit:
        indices = indices[: args.limit]
    my_indices = indices[rank::world]
    print(f"[eval] rank={rank} -> {len(my_indices)} items", flush=True)
    examples = benchmark_rows_to_examples(rows, indices=my_indices)
    for e in examples:
        m = e["meta"]
        e.update(idx=m["idx"], category=m["category"],
                 subcategory=m["subcategory"], difficulty=m["difficulty"],
                 gold_letter=m["gold_letter"], options=[None] * m["n_options"])
    if args.prompt_style == "tsrbench":
        from src.datasets_utils.prompts import user_text_tsrbench
        for e in examples:
            # Rebuild the user turn in the TSRBench JSON-with-reasoning format.
            r = rows[e["idx"]]
            markers = "<ts><ts/>" * len(e["ts_list"])
            e["system"] = None
            e["user_text"] = f"{markers}\n" + user_text_tsrbench(str(r["question"]), list(r["options"]))

    model, processor = load_model(
        args.model, args.adapter, device,
        dino_id=args.dino_model_id,
        dino_image_size=args.dino_image_size,
        dino_deepstack_indexes=tuple(int(x) for x in args.dino_deepstack_indexes.split(",")),
        prior_adapters=args.prior_adapter,
    )
    collator = UnifiedDualCollator(
        processor=processor, max_length=args.max_length,
        delay_image_size=args.dino_image_size,
        channel_stats=load_channel_stats(args, "eval"), chart_style=args.chart_style,
    )

    if args.prompt_style == "tsrbench" and args.max_new_tokens < 256:
        args.max_new_tokens = 512
    extract_fn = extract_letter_from_json if args.prompt_style == "tsrbench" else extract_letter

    miss_log_path = args.out.replace(".json", "_parse_misses.jsonl")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.time()
    for ex in tqdm(examples, desc=f"rank{rank}", disable=(rank != 0)):
        try:
            raw = run_one(model, processor, ex, collator, args.max_new_tokens)
        except Exception as e:
            raw = f"<<error:{type(e).__name__}: {e}>>"
        pred = extract_fn(raw, n_options=len(ex["options"]))
        rec = {
            "idx": ex["idx"],
            "category": ex["category"],
            "subcategory": ex["subcategory"],
            "difficulty": ex["difficulty"],
            "gold": ex["gold_letter"],
            "pred": pred,
            "raw": raw,
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
    n_parse_miss = sum(r["pred"] is None for r in records)
    report = {
        "benchmark": "tsexam",
        "model": args.model,
        "adapter": args.adapter,
        "prior_adapter": args.prior_adapter,
        "prompt_style": args.prompt_style,
        "chart_style": args.chart_style,
        "n": n,
        "correct": n_correct,
        "accuracy": n_correct / n if n else 0.0,
        "parse_misses": n_parse_miss,
        "elapsed_seconds": time.time() - t0,
        "provenance": provenance(args, data_json=args.data_json,
                                 benchmark_revision=(None if args.data_json else BENCHMARK_REVISION),
                                 limit=args.limit, world_size=world),
        "per_category": per_group_accuracy(records, "category"),
        "per_subcategory": per_group_accuracy(records, "subcategory"),
        "per_difficulty": per_group_accuracy(records, "difficulty"),
        "records": records,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[eval] DONE  acc={report['accuracy']:.4f}  ({n_correct}/{n})  "
          f"parse_misses={n_parse_miss}  -> {args.out}")
    for cat, stats in report["per_category"].items():
        print(f"  {cat:30s}  {stats['correct']:3d}/{stats['n']:3d}  acc={stats['acc']:.3f}")


if __name__ == "__main__":
    main()
