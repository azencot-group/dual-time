"""ChatTS-benchmark evaluation for the dual-tower model.

ChatTS questions carry inline `<ts><ts/>` markers, one per time series; they
go through UnifiedDualCollator exactly like ChatTS rows at training time.

Eval datasets (Zenodo doi:10.5281/zenodo.14349206), downloaded on first use
into eval_data/chatts_eval/ (`--data-dir`):
  - dataset_a.json : 159 univariate real-world series (NAB/Weather/Oracle/AIOps)
  - dataset_b.json : 400 multivariate (7-channel) synthetic series

Scoring is ChatTS's rule-based metric (categorical + numerical, no API key);
the RAGAS-backed 'reason' score is stubbed to 0.0 except the rule-based
`causal` ability. See scorer/scoring.py (the ChatTS authors' code, vendored
unchanged).

Usage (one process per GPU; items are sharded across ranks):
  torchrun --nproc_per_node=2 -m src.evals.chatts.eval --dataset both --model <base> \
      --prior-adapter <stage A> --adapter <stage B> --exp-label myrun
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src import paths
from src.evals.common import (DEFAULT_MODEL, add_determinism_arg, add_dual_model_args,
                              build_dual_collator, enable_determinism, greedy_cfg, init_distributed,
                              load_channel_stats, load_model_from_args, provenance, strip_think)
from .scorer.scoring import evaluate_batch_qa

ZENODO = "https://zenodo.org/api/records/14349206/files/{name}/content"
DEFAULT_DATA_DIR = paths.resolve("eval_data/chatts_eval")
DEFAULT_OUT_ROOT = paths.resolve("results/chatts_eval")
TS_MARKER = "<ts><ts/>"


def ensure_dataset(name: str, data_dir: str = DEFAULT_DATA_DIR) -> Path:
    """Return the local path to dataset_{a,b}.json under `data_dir`,
    downloading it if absent."""
    fname = f"dataset_{name}.json"
    data_dir = Path(data_dir)
    path = data_dir / fname
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        url = ZENODO.format(name=fname)
        print(f"[chatts-eval] downloading {fname} from Zenodo ...", flush=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def chatts_to_unified(sample: dict) -> dict:
    """ChatTS eval sample -> unified example (question already carries the
    markers; system is None, matching the training rows)."""
    ts_list = [np.asarray(ch, dtype=np.float32).ravel() for ch in sample["timeseries"]]
    return {"system": None, "user_text": sample["question"],
            "target_text": "", "ts_list": ts_list}


def run_one_unified(model, tokenizer, collator, unified_ex: dict, max_new_tokens: int):
    """Generate one answer. Returns (text, prompt_token_count)."""
    inputs = collator([unified_ex], for_generation=True)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=greedy_cfg(max_new_tokens))
    prompt_len = inputs["input_ids"].shape[1]
    gen = out[:, prompt_len:]
    return strip_think(tokenizer.batch_decode(gen, skip_special_tokens=False)[0]), int(prompt_len)


def evaluate_dataset(name: str, args, model, processor, collator, exp_label: str,
                     rank: int, world: int):
    """Run inference + scoring for one ChatTS dataset (a/b, or --dataset-json)."""
    src = Path(args.dataset_json) if getattr(args, "dataset_json", None) else ensure_dataset(name, args.data_dir)
    dataset = json.load(open(src))
    if args.limit:
        dataset = dataset[: args.limit]
    exp = f"{exp_label}_dataset_{name}"
    out_dir = Path(args.out_root) / exp
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    my_idx = list(range(rank, len(dataset), world))
    print(f"[chatts-eval] dataset_{name} [{exp_label}]: {len(dataset)} samples, "
          f"rank {rank} -> {len(my_idx)} items", flush=True)

    records = []
    t0 = time.time()
    for idx in tqdm(my_idx, desc=f"rank{rank} dataset_{name}", disable=(rank != 0)):
        sample = dataset[idx]
        try:
            response, n_tokens = run_one_unified(
                model, processor.tokenizer, collator, chatts_to_unified(sample),
                args.max_new_tokens)
        except Exception as e:
            response, n_tokens = f"<<error:{type(e).__name__}: {e}>>", 0
        records.append({
            "idx": idx,
            "question_text": sample["question"],
            "response": response,
            "num_tokens": n_tokens,
        })

    if world > 1:
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, records)
        if rank != 0:
            return None
        records = [r for sub in gathered for r in sub]

    records.sort(key=lambda r: r["idx"])
    json.dump(records, open(out_dir / "generated_answer.json", "wt"),
              ensure_ascii=False, indent=4)

    summary = evaluate_batch_qa(dataset, records, exp, num_workers=16, out_dir=str(out_dir),
                                normalize_enumeration=args.lenient_enumeration)
    summary["dataset"] = f"dataset_{name}"
    summary["exp"] = exp_label
    summary["adapter"] = args.adapter
    summary["n_samples"] = len(dataset)
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    summary["provenance"] = provenance(args, dataset_file=str(src), limit=args.limit, world_size=world,
                                       lenient_enumeration=bool(args.lenient_enumeration))
    print(f"[chatts-eval] dataset_{name} DONE  "
          f"categorical={summary['overall_categorical']}  "
          f"numerical={summary['overall_numerical']}  -> {out_dir}", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["a", "b", "both"], default="both")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help="Directory holding dataset_{a,b}.json (downloaded there if absent).")
    p.add_argument("--dataset-json", default=None,
                   help="Evaluate an arbitrary ChatTS-format dataset JSON (list of {timeseries, "
                        "cols, question, answer, attributes, ability_types}) instead of "
                        "dataset_{a,b} (README.md).")
    p.add_argument("--dataset-name", default="custom",
                   help="Name used in the output dir / summary for --dataset-json (e.g. my_eval_set).")
    p.add_argument("--exp-label", default=None,
                   help="Output subdir prefix; defaults to the adapter's parent dir name.")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--limit", type=int, default=None, help="Cap samples per dataset (quick check).")
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="Generation budget for the free-form numbered answer.")
    p.add_argument("--max-length", type=int, default=6144)
    p.add_argument("--lenient-enumeration", action="store_true",
                   help="Rewrite '(N)' list markers to 'N.' before scoring (a leniency beyond the "
                        "authors' protocol: can only raise a score). Default off = upstream parity.")
    add_dual_model_args(p, DEFAULT_MODEL)
    # Bit-exact mode is opt-in here: the generation is long and the
    # deterministic kernels are ~2x slower.
    add_determinism_arg(p, default=False)
    args = p.parse_args()
    if args.deterministic:
        enable_determinism()   # before the first CUDA call

    rank, world, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"[chatts-eval] rank={rank}/{world} device={device} "
          f"model={args.model} adapter={args.adapter}", flush=True)

    model, processor = load_model_from_args(args, device)
    collator = build_dual_collator(args, processor, load_channel_stats(args, "chatts-eval"))

    if args.exp_label:
        exp_label = args.exp_label
    elif args.adapter:
        exp_label = Path(args.adapter).parent.name   # .../stageB_dual8b/best -> stageB_dual8b
    else:
        exp_label = "base"

    out_root = Path(args.out_root)
    names = ["a", "b"] if args.dataset == "both" else [args.dataset]
    if args.dataset_json:
        names = [args.dataset_name]
    summaries = []
    for name in names:
        s = evaluate_dataset(name, args, model, processor, collator, exp_label, rank, world)
        if s is not None:
            summaries.append(s)

    if rank == 0 and summaries:
        # custom datasets (--dataset-json) get their own summary file so the A/B one survives
        sfile = out_root / (f"summary_{exp_label}_{args.dataset_name}.json" if args.dataset_json else f"summary_{exp_label}.json")
        json.dump(summaries, open(sfile, "wt"), indent=2)
        print(f"\n[chatts-eval] ===== SUMMARY ({exp_label}) =====")
        for s in summaries:
            print(f"  {s['dataset']:12s} n={s['n_samples']:4d}  "
                  f"categorical={s['overall_categorical']}  "
                  f"numerical={s['overall_numerical']}  "
                  f"reason[causal]={s['overall_reason_rulebased']}")
        print(f"[chatts-eval] summary -> {sfile}")


if __name__ == "__main__":
    main()
