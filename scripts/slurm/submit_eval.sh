#!/usr/bin/env bash
# Submit one benchmark (or all three) as a Slurm job: `python scripts/eval.py`
# inside the job, same flags.
#
#   scripts/slurm/submit_eval.sh --kind tsrb --stage-a DIR --stage-b DIR --label NAME
#   scripts/slurm/submit_eval.sh --kind all --dep afterok:<JOBID> --gpus 2 --time 06:00:00
#
# Job options: --gpus N (2) --time HH:MM:SS (tsexam 00:45:00, tsrb 03:55:00, chatts 01:30:00,
#              all 06:00:00; measured on 2x B200 with generous margins) --mem MEM (96G)
#              --cpus N (8) --dep <dependency> --test-only
#              --account/--partition/--qos (override scripts/slurm/slurm.env)
# Evaluation options, forwarded to scripts/eval.py (see `python scripts/eval.py --help`):
#              --kind (required) --stage-a --stage-b --no-stage-b --label --model --results-dir
#              --nproc (default: the job's GPUs) --limit --chart-style
#              --deterministic/--no-deterministic --lenient-enumeration
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIND=""; LABEL=""; GPUS=2; TIME=""; MEM="96G"; CPUS=8; DEP=""; SUBMIT=(); EVAL_ARGS=()
while [[ $# -gt 0 ]]; do case "$1" in
  --kind) KIND="$2"; shift 2 ;; --label) LABEL="$2"; EVAL_ARGS+=(--label "$2"); shift 2 ;;
  --gpus) GPUS="$2"; shift 2 ;; --time) TIME="$2"; shift 2 ;; --mem) MEM="$2"; shift 2 ;;
  --cpus) CPUS="$2"; shift 2 ;; --dep) DEP="$2"; shift 2 ;;
  --test-only) SUBMIT+=(--test-only); shift ;;
  --account|--partition|--qos) SUBMIT+=("$1" "$2"); shift 2 ;;
  --no-stage-b|--deterministic|--no-deterministic|--lenient-enumeration) EVAL_ARGS+=("$1"); shift ;;
  --stage-a|--stage-b|--model|--results-dir|--nproc|--limit|--chart-style) EVAL_ARGS+=("$1" "$2"); shift 2 ;;
  *) echo "unknown arg $1"; exit 2 ;;
esac; done
case "$KIND" in
  tsexam) TIME="${TIME:-00:45:00}" ;;
  tsrb)   TIME="${TIME:-03:55:00}" ;;
  chatts) TIME="${TIME:-01:30:00}" ;;
  all)    TIME="${TIME:-06:00:00}" ;;
  *) echo "usage: $0 --kind tsexam|tsrb|chatts|all [--stage-a DIR --stage-b DIR --label NAME ...]"; exit 2 ;;
esac
[[ " ${EVAL_ARGS[*]} " == *" --nproc "* ]] || EVAL_ARGS+=(--nproc "$GPUS")
DEPA=(); [[ -n "$DEP" ]] && DEPA=(-d "$DEP")
exec "$HERE/submit_cmd.sh" -n "eval-${LABEL:-ckpt}-$KIND" -g "$GPUS" -t "$TIME" -m "$MEM" -c "$CPUS" \
  "${DEPA[@]}" "${SUBMIT[@]}" -- "python scripts/eval.py --kind $KIND ${EVAL_ARGS[*]}"
