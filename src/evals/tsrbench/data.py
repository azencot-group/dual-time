"""Load TSRBench items as *unified* examples for the dual-tower model.

TSRBench (https://tsrbench.github.io) is a multi-task multiple-choice benchmark
over time series. Every item carries a `timeseries` (a list of 1-D series), a
`question`, a set of `choices` (list, letter-keyed dict, or — for `perception` —
inline in the question text), and a single-letter `answer`.

We convert each item into the same unified-example schema the dual collator
consumes (see `chatts_data.py` / `unified_collator.UnifiedDualCollator`):

    {system, user_text (with <ts><ts/> markers), target_text,
     ts_list, ts_field_names, task, meta}

One `<ts><ts/>` marker per series -> the collator renders, *per series*, a
matplotlib chart (native Qwen ViT, image stream) AND a delay-embedding image
(DINOv3, video stream). That is exactly the dual-tower input the unified model
was trained on.

Prompting mirrors the TSExam letter convention the unified model was trained on
(`prompts.SYSTEM` + question + lettered options, gold = a single letter) so the
model answers in its native single-letter format. Pass `prompt_style="tsrbench"`
to instead use the official TSRBench JSON-reasoning prompt (parsed with
`evals.common.extract_letter_from_json`).

Out-of-distribution note: TSRBench series carry arbitrary amplitudes that are
not in the training delay global-norm stats, so their `ts_field_names` will not
match any stats key -> the collator falls back to per-series instance norm,
which is the principled choice for the (shape-based) delay embedding.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Sequence

import numpy as np

from src import paths
from src.datasets_utils.prompts import LETTERS, SYSTEM, format_options, user_text_tsrbench

DEFAULT_DATASET_DIR = paths.resolve("eval_data/tsrbench")

TS_MARKER = "<ts><ts/>"

# task -> top-level TSRBench capability group (for grouped reporting)
TASK_GROUP = {
    "perception": "perception",
    "etiological_reasoning": "reasoning",
    "causal_reasoning": "reasoning",
    "abductive_reasoning": "reasoning",
    "temporal_relation_reasoning": "reasoning",
    "numerical_reasoning": "reasoning",
    "deductive_reasoning": "reasoning",
    "inductive_reasoning": "reasoning",
    "time_series_forecasting": "prediction",
    "event_prediction": "prediction",
    "qualitative_decision": "decision",
    "quantitative_decision": "decision",
}

# task -> jsonl path relative to the dataset root
TASK_FILE = {
    "perception": "perception/perception.jsonl",
    "etiological_reasoning": "reasoning/etiological_reasoning.jsonl",
    "causal_reasoning": "reasoning/causal_reasoning.jsonl",
    "abductive_reasoning": "reasoning/abductive_reasoning.jsonl",
    "temporal_relation_reasoning": "reasoning/temporal_relation_reasoning.jsonl",
    "numerical_reasoning": "reasoning/numerical_reasoning.jsonl",
    "deductive_reasoning": "reasoning/deductive_reasoning.jsonl",
    "inductive_reasoning": "reasoning/inductive_reasoning.jsonl",
    "time_series_forecasting": "prediction/time_series_forecasting.jsonl",
    "event_prediction": "prediction/event_prediction.jsonl",
    "qualitative_decision": "decision/qualitative_decision.jsonl",
    "quantitative_decision": "decision/quantitative_decision.jsonl",
}

ALL_TASKS = list(TASK_FILE.keys())
# Default eval set excludes abductive (its series live in a different
# `numerical_time_series` dict and the questions are basketball-win-prob
# specific). It is supported via `_abductive_series` but opt-in.
DEFAULT_TASKS = [t for t in ALL_TASKS if t != "abductive_reasoning"]


# --------------------------------------------------------------------------- #
# series / choice helpers
# --------------------------------------------------------------------------- #
# Series named like a time axis (or carrying datetime strings) are dropped —
# they are not plottable values. Matches the original TSRBench renderer, which
# skips 'Time Stamps' (temporal) and 'Time' (etiological).
_TIME_AXIS_NAMES = {
    "time", "times", "time stamp", "time stamps", "timestamp", "timestamps",
    "date", "dates", "datetime", "index",
}


def _try_float(s) -> Optional[np.ndarray]:
    """Convert a series to a 1-D float32 array, or None if it isn't numeric
    (e.g. a column of datetime strings)."""
    try:
        return np.asarray(s, dtype=np.float32).ravel()
    except (ValueError, TypeError):
        return None


def _series_and_names(timeseries, names, task: str):
    """Normalise `timeseries` to a list of 1-D float arrays plus parallel field
    names, dropping any time-axis series (non-numeric or time-named).

    TSRBench stores `timeseries` as a list of series (each a list of numbers);
    a flat list of scalars is treated as a single series. `name_of_series`,
    when present and length-matched, gives per-series names.
    """
    if timeseries is None:
        return [], []
    seq = list(timeseries)
    if not seq:
        return [], []
    raw = list(seq) if isinstance(seq[0], (list, tuple, np.ndarray)) else [seq]
    name_list = ([str(n) for n in names] if names and len(names) == len(raw)
                 else [None] * len(raw))

    out_ts: List[np.ndarray] = []
    out_names: List[str] = []
    for i, s in enumerate(raw):
        nm = name_list[i]
        if nm is not None and nm.strip().lower() in _TIME_AXIS_NAMES:
            continue
        arr = _try_float(s)
        if arr is None:          # datetime/string axis -> not a value series
            continue
        out_ts.append(arr)
        out_names.append(nm if nm is not None else f"{task}_s{len(out_ts) - 1}")
    return out_ts, out_names


def _parse_choice_series(choices) -> Optional[List[np.ndarray]]:
    """Parse a letter-keyed `choices` dict of numeric sequences (raw lists or
    stringified `[..]`) into a list of 1-D float arrays in A,B,C,… order."""
    if not isinstance(choices, dict):
        return None
    import ast
    keys = sorted(choices.keys(), key=lambda k: LETTERS.index(str(k).upper()))
    out = []
    for k in keys:
        v = choices[k]
        if isinstance(v, str):
            try: v = ast.literal_eval(v.strip())
            except Exception: return None
        try: arr = np.asarray(v, dtype=np.float32).ravel()
        except Exception: return None
        if arr.size == 0: return None
        out.append(arr)
    return out


def _choices_are_series(item, n_ts: int) -> bool:
    """True when the answer choices ARE the time series (e.g. forecasting: pick the
    candidate trajectory). Signalled by a letter-keyed `choices` dict whose values
    are numeric sequences (raw lists or stringified `[..]`), one per encoded series."""
    ch = item.get("choices")
    if not isinstance(ch, dict) or len(ch) < 2 or len(ch) != n_ts:
        return False
    def _is_seq(v):
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            return True
        if isinstance(v, str):
            s = v.strip()
            return s.startswith("[") and "," in s
        return False
    return all(_is_seq(v) for v in ch.values())


def _choices_to_list(choices) -> Optional[List[str]]:
    """Return choices as an ordered list of stringified option contents, or
    None when the item has no explicit choices (perception: inline in question).

    Letter-keyed dicts (`{"A": ..., "B": ...}`) are ordered by letter so the
    dataset's gold letter still indexes the right option; other dicts are
    ordered by sorted key.
    """
    if choices is None:
        return None
    if isinstance(choices, dict):
        keys = list(choices.keys())
        if all(str(k).upper() in LETTERS for k in keys):
            keys = sorted(keys, key=lambda k: LETTERS.index(str(k).upper()))
        else:
            keys = sorted(keys, key=str)
        return [_stringify(choices[k]) for k in keys]
    if isinstance(choices, (list, tuple)):
        return [_stringify(c) for c in choices]
    return None


def _stringify(v) -> str:
    """Compact string form of an option value (may be text, a number, or a
    nested list such as a forecast window or an adjacency matrix)."""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(v)


def _infer_n_options(question: str, choices_list: Optional[List[str]]) -> int:
    """Number of options. From `choices` when present, else parse inline option
    letters out of the question text (perception), else default to 4."""
    if choices_list:
        return len(choices_list)
    letters = set(re.findall(r"(?m)^\s*([A-H])[.)]\s+", question))
    if letters:
        return max(LETTERS.index(c) + 1 for c in letters)
    return 4


def _place_markers(question: str, n_series: int) -> str:
    """Ensure the question carries exactly `n_series` <ts><ts/> markers.

    If the question already has the right number (e.g. perception embeds one
    marker inline), keep it; otherwise strip any stray markers and prepend a
    fresh block (one marker per series, the chatts_data convention).
    """
    have = question.count(TS_MARKER)
    if have == n_series:
        return question
    q = question.replace(TS_MARKER, "").strip()
    return (TS_MARKER * n_series) + "\n" + q


# --------------------------------------------------------------------------- #
# per-item conversion
# --------------------------------------------------------------------------- #
def _abductive_series(item, ar_context: bool = False,
                      ar_all_channels: bool = False,
                      ar_full_series: bool = False) -> tuple[List[np.ndarray], List[str], str, List[str], str]:
    """Adapter for abductive_reasoning items.

    Series live in `numerical_time_series` as {name: {history: [...], future:
    [...]}}. The question/options/gold come from `multiple_choice_question`.
    """
    nts = item.get("numerical_time_series", {}) or {}
    # Protocol: the question asks about "data provided up to this point"; the
    # official harness cuts at the critical moment and keys on the
    # win-probability signal. Default = WP channels only + HISTORY only;
    # `ar_all_channels` / `ar_full_series` (--ar-all-channels / --ar-full-series
    # on the evaluator) show everything instead.
    _wp_only = not ar_all_channels
    _hist_only = not ar_full_series
    ts_list, names = [], []
    for name, parts in nts.items():
        if _wp_only and not str(name).lower().startswith("wp"):
            continue
        if isinstance(parts, dict):
            seq = list(parts.get("history", []))
            if not _hist_only:
                seq += list(parts.get("future", []))
        else:
            seq = list(parts)
        if seq:
            ts_list.append(np.asarray(seq, dtype=np.float32).ravel())
            names.append(str(name))
    mcq = item.get("multiple_choice_question", {}) or {}
    question = str(mcq.get("question", ""))
    # `ar_context` (src.evals.tsrbench.eval passes True for this task): prepend
    # the game context exactly as the official TSRBench harness does (past
    # events, a critical-moment marker, future events). Without it the model
    # only sees two win-probability curves and a bare question, which scores
    # at chance; with it the reference checkpoint scores 56.0.
    if ar_context:
        ctx = item.get("context", {}) or {}
        past = "\n".join(f"- {t}: {e}" for t, e in zip(ctx.get("history_times", []), ctx.get("history_events", [])))
        fut = "\n".join(f"- {t}: {e}" for t, e in zip(ctx.get("future_times", []), ctx.get("future_events", [])))
        n_hist = (len((nts.get("wp_Team A") or {}).get("history", []))
                  if isinstance(nts.get("wp_Team A"), dict) else None)
        series_note = ("The win-probability series cover the whole game segment (past and future events); "
                       f"the critical moment is at index {n_hist}." if not _hist_only and n_hist is not None
                       else "The win-probability series are provided up to the critical moment.")
        question = ("You are an expert in basketball game analysis. Your task is to perform abductive reasoning. "
                    "Given a sequence of past events, future events, and the corresponding win-probability time series, "
                    "determine the most plausible event that occurred in between to link them.\n"
                    f"Past Events (History):\n{past}\n... [A CRITICAL EVENT HAPPENED HERE] ...\nFuture Events:\n{fut}\n"
                    f"{series_note}\n{question}")
    choices_list = _choices_to_list(mcq.get("choices"))
    gold = str(mcq.get("answer", "")).strip().upper()[:1]
    return ts_list, names, question, (choices_list or []), gold


def tsrbench_item_to_unified(item: dict, task: str, idx: int,
                             prompt_style: str = "qa",
                             series_choices_inline: bool = False,
                             series_choices_keep_text: bool = False,
                             ar_context: bool = False,
                             ar_all_channels: bool = False,
                             ar_full_series: bool = False) -> Optional[dict]:
    """Convert one TSRBench item into a unified example, or None if unusable.

    `series_choices_inline`: when the answer choices ARE the series (deductive
    reasoning and forecasting), render each option as an inline `A) <ts><ts/>`
    marker (one encoded series per option) and DROP the stringified-number-list
    option text, so the model reads the candidate series through the encoders,
    not as raw numbers. This is the layout the model was trained on. For
    forecasting the inline layout scores badly (the four candidates are
    permutations of one continuation and the model cannot tell them apart from
    small charts), so the reference protocol re-scores forecasting WITHOUT this
    flag (plain text candidates, history still encoded) at a 16k token budget
    (README.md; src.evals.tsrbench.eval applies the protocol per task).
    `ar_context`: abductive items get the harness's game context;
    `ar_all_channels` / `ar_full_series`: see `_abductive_series`.
    `series_choices_keep_text`: in the inline path, ALSO append the text numbers after
    each marker -> `A) <ts><ts/> [0.52, 0.57, ...]` (both encoded image AND raw text).
    """
    if task == "abductive_reasoning":
        ts_list, field_names, question, choices_list, gold = _abductive_series(
            item, ar_context=ar_context, ar_all_channels=ar_all_channels, ar_full_series=ar_full_series)
    else:
        ts_list, field_names = _series_and_names(
            item.get("timeseries"), item.get("name_of_series"), task)
        question = str(item.get("question", ""))
        choices_list = _choices_to_list(item.get("choices"))
        gold = str(item.get("answer", "")).strip().upper()[:1]

    if not ts_list or gold not in LETTERS:
        return None

    # Series-as-choices: encode each candidate OPTION series (from `choices`) as an
    # inline A) <ts><ts/> marker, no text number-dump. The option series come
    # from `choices` (NOT `timeseries`): for forecasting `timeseries` is the
    # history, kept as separate context markers; for deductive reasoning
    # `timeseries` == `choices` (no extra context).
    # ts_list = [context..., optionA, optionB, ...] matching markers.
    if (series_choices_inline and task != "abductive_reasoning"
            and _choices_are_series(item, len(ts_list))):
        opt_series = _parse_choice_series(item.get("choices"))
        if opt_series is not None:
            n_options = len(opt_series)
            ctx = ts_list  # from _series_and_names(timeseries)
            same = (len(ctx) == n_options and all(
                ctx[i].shape == opt_series[i].shape and np.allclose(ctx[i], opt_series[i])
                for i in range(n_options)))
            # Training-parity rendering: byte-matches the layout
            # reasoning_data.reasoning_corpus_rows_to_unified emits for shape (C)
            # rows: context markers prefix, then the question, then a
            # 'Candidate series (pick the matching one):' block with one
            # 'A) <ts><ts/>' line per option. Candidate field names are
            # 'choice_<L>' (drives the collator's per-series-chart guard and
            # the same 'choice_A:' text labels the model saw in training).
            new_ts, new_names = [], []
            ctx_block = ""
            if not same:  # forecasting: timeseries is separate history context
                # History labeling: training always labels the history series
                # in TEXT ('Signal: <ts>'); TSRBench TSF items carry ONE name
                # for 4 multivariate channels, so the raw names are placeholders
                # and the labeling pass would leave the history charts
                # anonymous. Label them 'history channel N' (or the real name
                # when per-channel names exist) so the model can tell history
                # apart from the candidate charts.
                for j, s in enumerate(ctx):
                    nm = field_names[j] if j < len(field_names) else ""
                    if not nm or re.fullmatch(r"\w+_s\d+", nm):
                        nm = f"history channel {j + 1}" if len(ctx) > 1 else "history"
                    new_ts.append(s)
                    new_names.append(nm)
                ctx_block = (TS_MARKER * len(ctx)) + "\n"
            ch_txt = _choices_to_list(item.get("choices")) or []
            opt_block = ""
            for i in range(n_options):
                new_ts.append(opt_series[i]); new_names.append(f"choice_{LETTERS[i]}")
                if series_choices_keep_text and i < len(ch_txt):
                    opt_block += f"{LETTERS[i]}) {TS_MARKER} {ch_txt[i]}\n"
                else:
                    opt_block += f"{LETTERS[i]}) {TS_MARKER}\n"
            q = question.replace(TS_MARKER, "").strip()
            cand_block = f"\n\nCandidate series (pick the matching one):\n{opt_block}"
            if prompt_style == "tsrbench":
                user = (f"{ctx_block}{q}{cand_block}\n"
                        "Output your reasoning and answer in JSON format.")
                system = None
            else:
                user = f"{ctx_block}{q}{cand_block}\nAnswer:"
                system = SYSTEM
            return {
                "system": system, "user_text": user, "target_text": gold,
                "ts_list": new_ts, "ts_field_names": new_names, "task": "tsrbench",
                "meta": {"idx": idx, "task": task, "group": TASK_GROUP.get(task, "other"),
                         "gold_letter": gold, "n_options": n_options,
                         "series_choices_inline": True,
                         "n_context_series": (0 if same else len(ctx))},
            }

    # TSF history labeling also applies in the PLAIN (text-options) path: the
    # 4 multivariate history channels carry placeholder names, which the v2
    # text-labeling pass would skip, leaving the history charts anonymous.
    if task == "time_series_forecasting":
        field_names = [
            (nm if nm and not re.fullmatch(r"\w+_s\d+", str(nm))
             else (f"history channel {j + 1}" if len(ts_list) > 1 else "history"))
            for j, nm in enumerate(field_names)]

    n_options = _infer_n_options(question, choices_list)
    q_with_markers = _place_markers(question, len(ts_list))

    if prompt_style == "tsrbench":
        # Official TSRBench prompt: no system role, JSON-with-reasoning output.
        opts = choices_list or []
        body = user_text_tsrbench(q_with_markers, opts) if opts else \
            (q_with_markers + "\n\nOutput your reasoning and answer in JSON format.")
        system = None
        user = body
    else:  # "qa" — the TSExam single-letter convention the model was trained on
        system = SYSTEM
        if choices_list:
            user = f"{q_with_markers}\nOptions:\n{format_options(choices_list)}\nAnswer:"
        else:
            # perception etc.: options already inline in the question text
            user = f"{q_with_markers}\nAnswer:"

    return {
        "system": system,
        "user_text": user,
        "target_text": gold,
        "ts_list": ts_list,
        "ts_field_names": field_names,
        "task": "tsrbench",
        "meta": {
            "idx": idx,
            "task": task,
            "group": TASK_GROUP.get(task, "other"),
            "gold_letter": gold,
            "n_options": n_options,
        },
    }


# --------------------------------------------------------------------------- #
# dataset loading
# --------------------------------------------------------------------------- #
def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tsrbench_task(dataset_dir: str, task: str, prompt_style: str = "qa",
                       limit: Optional[int] = None,
                       series_choices_inline: bool = False,
                       series_choices_keep_text: bool = False,
                       ar_context: bool = False,
                       ar_all_channels: bool = False,
                       ar_full_series: bool = False) -> List[dict]:
    """Load one TSRBench task file as unified examples (with global idx local to
    the task; the caller renumbers across tasks)."""
    path = os.path.join(dataset_dir, TASK_FILE[task])
    rows = _read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    out = []
    for i, r in enumerate(rows):
        ex = tsrbench_item_to_unified(r, task, i, prompt_style=prompt_style,
                                      series_choices_inline=series_choices_inline,
                                      series_choices_keep_text=series_choices_keep_text,
                                      ar_context=ar_context,
                                      ar_all_channels=ar_all_channels, ar_full_series=ar_full_series)
        if ex is not None:
            out.append(ex)
    return out


def load_tsrbench_unified(dataset_dir: str, tasks: Optional[Sequence[str]] = None,
                          prompt_style: str = "qa",
                          limit_per_task: Optional[int] = None,
                          series_choices_inline: bool = False,
                          series_choices_keep_text: bool = False,
                          ar_context: bool = False,
                          ar_all_channels: bool = False,
                          ar_full_series: bool = False) -> List[dict]:
    """Load the requested TSRBench tasks as unified examples.

    A global `meta['idx']` is assigned across all tasks (stable sort key for the
    multi-rank gather in src.evals.tsrbench.eval). `tasks=None` -> DEFAULT_TASKS.
    """
    tasks = list(tasks) if tasks else list(DEFAULT_TASKS)
    out: List[dict] = []
    gid = 0
    for task in tasks:
        if task not in TASK_FILE:
            raise ValueError(f"unknown TSRBench task: {task!r}")
        rows = load_tsrbench_task(dataset_dir, task, prompt_style=prompt_style,
                                  limit=limit_per_task,
                                  series_choices_inline=series_choices_inline,
                                  series_choices_keep_text=series_choices_keep_text,
                                  ar_context=ar_context,
                                  ar_all_channels=ar_all_channels, ar_full_series=ar_full_series)
        for ex in rows:
            ex["meta"]["idx"] = gid
            gid += 1
        print(f"[tsrbench_data] {task:30s} -> {len(rows)} examples", flush=True)
        out.extend(rows)
    print(f"[tsrbench_data] total unified examples: {len(out)}", flush=True)
    return out


if __name__ == "__main__":  # quick check (CPU-only: builds examples, no model)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    p.add_argument("--tasks", default=None, help="comma-separated; default = all but abductive")
    p.add_argument("--prompt-style", default="qa", choices=["qa", "tsrbench"])
    p.add_argument("--limit-per-task", type=int, default=2)
    a = p.parse_args()
    tasks = a.tasks.split(",") if a.tasks else None
    ex = load_tsrbench_unified(a.dataset_dir, tasks=tasks,
                               prompt_style=a.prompt_style,
                               limit_per_task=a.limit_per_task)
    for e in ex[:6]:
        print("task:", e["meta"]["task"], "n_ts:", len(e["ts_list"]),
              "markers:", e["user_text"].count(TS_MARKER),
              "gold:", e["target_text"], "n_opt:", e["meta"]["n_options"])
        print("  user head:", e["user_text"][:160].replace("\n", " "))
