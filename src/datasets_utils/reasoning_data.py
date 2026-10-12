"""Loader for the TSRBench-style reasoning corpus (`reasoning_corpus`): letter,
JSON-rationale ("cot") or thinking-token targets, per-operator filters.
"""

from __future__ import annotations

import json
import random
from typing import List, Optional

import numpy as np

from .common import TS_MARKER
from .mcq import LETTERS, _LETTER_REF_RE, _remap_answer_trailer, _shuffle_mcq, resolve_letter
from .prompts import SYSTEM as TSEXAM_SYSTEM, assistant_json_tsrbench, user_text_tsrbench


def reasoning_corpus_rows_to_unified(rows, indices=None, target_format="letter",
                                     only_rationale=False, reasoning_task=None,
                                     shuffle_options=False,
                                     shuffle_seed=0) -> List[dict]:
    """Convert TimeSeriesExam *reasoning-operators* rows into unified examples.

    Same MC schema as tsexam, but TR rows additionally carry a gold `rationale`
    (a short chain-of-thought). Two label formats:

      - "letter" : target = gold letter (system = TSEXAM letter prompt). Identical
                   to tsexam — the control for "do the examples help".
      - "cot"    : rows WITH a rationale are trained in the exact TSRBench
                   JSON-with-reasoning format that `eval_tsrbench --prompt-style
                   tsrbench` feeds (system=None, "A. " options, "Output your
                   reasoning and answer in JSON format."), target =
                   {"reasoning": <rationale>, "answer": <letter>}. Rows WITHOUT a
                   rationale fall back to the letter format.

    `only_rationale=True` keeps just the rows that carry a non-empty rationale
    (the 440 TR examples) so the CoT signal isn't diluted by the rows with no trace.

    `reasoning_task` (str or set of str, e.g. "TR") keeps only rows whose
    `reasoning_task` field matches — the robust way to isolate one operator family
    regardless of which rows happen to carry a rationale. Schema-tolerant: handles
    both the original reasoning_dataset.json (single `ts`, full-text `answer`) and
    the newer subset (text options + full-text `answer` + `rationale`); rows whose
    `answer` can't be resolved against `options` or that lack a series are skipped.
    """
    if indices is None:
        indices = range(len(rows))
    if isinstance(reasoning_task, str):
        reasoning_task = {reasoning_task}
    out: List[dict] = []
    for i in indices:
        r = rows[i]
        if reasoning_task and str(r.get("reasoning_task", "")) not in reasoning_task:
            continue
        rationale = str(r.get("rationale", "") or "").strip()
        if only_rationale and not rationale:
            continue
        try:
            options = list(r["options"])
            gold = resolve_letter(r["answer"], options)
        except Exception:
            continue
        if gold is None:
            continue
        # The real choice content can live in `choices` (a dict) rather than the
        # bare-letter `options`. Three shapes:
        #   (A) text options   : options[] hold the choice text        (AR/ER/TR)
        #   (B) text choices    : choices{letter:text}                  (NR/IR/CD: numbers/matrices)
        #   (C) series choices  : choices{letter:[floats]} -> pick the matching series (DR/TSF)
        # (C) values may also arrive as STRINGS ("[90.8, 94.2, ...]");
        # _choice_series() parses those too so candidate curves are rendered
        # as charts in training exactly as they are at eval.
        n_opt = len(options)
        letters = [LETTERS[j] for j in range(n_opt)]
        choices = r.get("choices")

        def _choice_series(v):
            """A numeric series from a choices[] value: list of numbers, or a
            string that json-parses to one. None otherwise."""
            if isinstance(v, str) and v.lstrip()[:1] == "[":
                try:
                    v = json.loads(v)
                except Exception:
                    return None
            if (isinstance(v, list) and len(v) > 1
                    and all(isinstance(x, (int, float)) for x in v[:8])):
                return np.asarray(v, dtype=np.float32).ravel()
            return None

        ch_series = None
        if isinstance(choices, dict) and choices and all(L in choices for L in letters):
            parsed = [_choice_series(choices[L]) for L in letters]
            if all(p is not None for p in parsed):
                ch_series = parsed
        ch_is_series = ch_series is not None
        ch_is_text = (isinstance(choices, dict) and not ch_is_series
                      and all(L in choices for L in letters))

        ts_list, ts_field_names = [], []
        # Context series (history / channels): ts1..tsN + legacy single "ts".
        # Included for ALL shapes — a TSF row has BOTH a history series and
        # series choices; dropping the history made the forecast unanswerable.
        # DR-style rows mirror the candidates into ts1..tsN as well: any
        # context series identical to a choice series is skipped (dedup).
        named = r.get("name_of_series") or []
        for idx_k, k in enumerate([f"ts{i}" for i in range(1, 17)] + ["ts"]):
            v = r.get(k)
            if v is not None and len(v):
                arr = np.asarray(v, dtype=np.float32).ravel()
                if ch_series is not None and any(
                        arr.shape == c.shape and np.allclose(arr, c) for c in ch_series):
                    continue
                ts_list.append(arr)
                nm = (named[len(ts_list) - 1]
                      if len(named) >= len(ts_list) and named[len(ts_list) - 1] else k)
                ts_field_names.append(str(nm))
        # AR gametime: the official tsrbench abductive protocol shows the
        # win-probability signal ONLY — drop the score channels so training
        # matches eval (same rule that produced ar_gametime_wp.json).
        if str(r.get("reasoning_task", "")) == "AR":
            wp = [j for j, nm in enumerate(ts_field_names)
                  if "win prob" in nm.lower() or nm.lower().startswith("wp")]
            if wp:
                ts_list = [ts_list[j] for j in wp]
                ts_field_names = [ts_field_names[j] for j in wp]
        n_ctx = len(ts_list)
        if ch_is_series:
            # (C) the candidates ARE series: context markers first (if any),
            # then one labeled marker per choice, in letter order
            for L, arr in zip(letters, ch_series):
                ts_list.append(arr)
                ts_field_names.append(f"choice_{L}")
            ctx_markers = "".join(TS_MARKER for _ in range(n_ctx))
            ctx_prefix = f"{ctx_markers}\n" if n_ctx else ""
            question_block = (f"{ctx_prefix}{r['question']}\n\nCandidate series (pick the matching one):\n"
                              + "".join(f"{L}) {TS_MARKER}\n" for L in letters))
            opt_texts = letters
        else:
            # (A)/(B): context series markers at front; choices come from
            # `choices` (B) or `options` (A)
            markers = "".join(TS_MARKER for _ in ts_list)
            question_block = f"{markers}\n{r['question']}"
            opt_texts = [str(choices[L]) for L in letters] if ch_is_text else [str(o) for o in options]
        if not ts_list:
            continue

        # Optional deterministic option shuffle. Never for shape (C): there the
        # letters are bound to marker positions. Skipped for rationales that
        # reference options by letter (they'd point at the wrong option).
        option_perm = None
        if (shuffle_options and not ch_is_series and n_opt > 1
                and not _LETTER_REF_RE.search(rationale)):
            old_gold = gold
            opt_texts, gold, option_perm = _shuffle_mcq(
                opt_texts, gold,
                # Seed string kept as "tsexam_reasoning" (the loader's former
                # name) so option permutations match the reported runs.
                random.Random(f"tsexam_reasoning:{shuffle_seed}:{i}"))
            rationale = _remap_answer_trailer(rationale, old_gold, gold)

        use_cot = (target_format == "cot" and rationale)
        # "think": native thinking-token format. The rationale goes in the
        # model's <think>..</think> block and the answer is the bare letter, so
        # the model learns to reason inside <think> then emit the choice. The
        # collator reads `reasoning_text` and renders the thinking turn; rows
        # without a rationale fall back to letter.
        use_think = (target_format == "think" and rationale)
        reasoning_text = None
        if use_cot:
            # TR only (rationale present, text options): byte-identical to
            # src/evals/tsrbench/data.py prompt_style="tsrbench" (verified round-trip).
            system = None
            user_text = user_text_tsrbench(question_block, opt_texts)
            target_text = assistant_json_tsrbench(rationale, gold)
        elif ch_is_series:
            system = TSEXAM_SYSTEM
            user_text = f"{question_block}\nAnswer:"
            target_text = gold
            if use_think:
                reasoning_text = rationale
        else:
            system = TSEXAM_SYSTEM
            user_text = f"{question_block}\nOptions:\n" + "\n".join(
                f"{letters[j]}) {opt_texts[j]}" for j in range(n_opt)) + "\nAnswer:"
            target_text = gold
            if use_think:
                reasoning_text = rationale
        tf = "cot" if use_cot else ("think" if use_think else "letter")
        ex = {
            "system": system,
            "user_text": user_text,
            "target_text": target_text,
            "ts_list": ts_list,
            "ts_field_names": ts_field_names,
            "task": "reasoning_corpus",
            "meta": {
                "idx": i,
                "id": r.get("id"),
                "tid": r.get("tid"),
                "reasoning_task": str(r.get("reasoning_task", "")),
                "subcategory": str(r.get("subcategory", "")),
                "difficulty": str(r.get("difficulty", "")),
                "gold_letter": gold,
                "n_options": len(options),
                "option_perm": option_perm,
                "has_rationale": bool(rationale),
                "target_format": tf,
            },
        }
        if reasoning_text:
            ex["reasoning_text"] = reasoning_text
        out.append(ex)
    return out


def load_reasoning_corpus(json_path: str, limit: Optional[int] = None,
                          target_format: str = "letter",
                          only_rationale: bool = False,
                          reasoning_task=None,
                          shuffle_options: bool = False,
                          shuffle_seed: int = 0) -> List[dict]:
    """Load a TimeSeriesExam reasoning-operators JSON as unified examples.

    `limit` is applied AFTER the `only_rationale` / `reasoning_task` filters.
    `reasoning_task` (e.g. "TR" or a comma string "TR,DR") selects one operator
    family; works on both the original and the newer subset schema.
    """
    with open(json_path) as f:
        rows = json.load(f)
    if isinstance(reasoning_task, str) and "," in reasoning_task:
        reasoning_task = {t.strip() for t in reasoning_task.split(",") if t.strip()}
    out = reasoning_corpus_rows_to_unified(
        rows, target_format=target_format, only_rationale=only_rationale,
        reasoning_task=reasoning_task,
        shuffle_options=shuffle_options, shuffle_seed=shuffle_seed)
    if limit is not None:
        out = out[:limit]
    return out
