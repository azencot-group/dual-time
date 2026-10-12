"""`load_unified_sources`: the config-driven entry point the trainer uses. Each
`unified_sources` entry names a loader (`type`) and its arguments; see the
docstring for the types.
"""

from __future__ import annotations

from typing import List

from .caption_data import load_chatts_style_captions, load_structural_captions
from .catsbench_data import load_catsbench
from .chatts_data import load_chatts
from .common import load_unified_jsonl
from .reasoning_data import load_reasoning_corpus
from .tsexam_data import load_numerical_corpus, load_tsexam_perception


def load_unified_sources(sources: List[dict]) -> List[dict]:
    """Load and concatenate several sources.

    `sources` is a list of dicts (every entry accepts `limit` and `task_tag`).
    Source names follow the paper's training-data table (Stage I: CaTS-Bench,
    ChatTS-style captions, structural captions; Stage II: ChatTS ift/sft, TSExam
    perception, numerical corpus, reasoning corpus):
      {"type": "catsbench",            "split": "train"|"test"}
      {"type": "chatts_style_captions","path": ".../chatts_style_captions.jsonl", "target_field": "caption"}
      {"type": "structural_captions",  "path": ".../structural_captions.json"}
      {"type": "chatts",               "subset": "ift"|"sft"|"align_256"|"dev"}
      {"type": "tsexam_perception",    "path": ".../tsexam_perception.json", "shuffle_options": bool?}
      {"type": "numerical_corpus",     "path": ".../numerical_corpus.json"}
      {"type": "reasoning_corpus",     "path": ".../reasoning_corpus.json", "target_format": "letter"|"cot"|"think",
                                       "reasoning_task": [..]?, "only_rationale": bool?, "shuffle_options": bool?}
      {"type": "jsonl",                "path": ".../rows.jsonl"}   # pre-built unified rows
    """
    out: List[dict] = []
    for s in sources:
        limit = s.get("limit")
        if s["type"] == "chatts":
            rows = load_chatts(s["subset"], limit=limit)
            print(f"[sources] chatts:{s['subset']} -> {len(rows)} examples", flush=True)
        elif s["type"] == "tsexam_perception":
            rows = load_tsexam_perception(
                s["path"], limit=limit,
                shuffle_options=bool(s.get("shuffle_options", False)),
                shuffle_seed=int(s.get("shuffle_seed", 0)))
            print(f"[sources] tsexam_perception:{s['path']} "
                  f"shuffle_options={bool(s.get('shuffle_options', False))} "
                  f"-> {len(rows)} examples", flush=True)
        elif s["type"] == "numerical_corpus":
            rows = load_numerical_corpus(s["path"], limit=limit)
            print(f"[sources] numerical_corpus:{s['path']} -> {len(rows)} examples", flush=True)
        elif s["type"] == "reasoning_corpus":
            rows = load_reasoning_corpus(
                s["path"], limit=limit,
                target_format=s.get("target_format", "letter"),
                only_rationale=bool(s.get("only_rationale", False)),
                reasoning_task=s.get("reasoning_task"),
                shuffle_options=bool(s.get("shuffle_options", False)),
                shuffle_seed=int(s.get("shuffle_seed", 0)),
            )
            n_cot = sum(1 for r in rows if r["meta"].get("target_format") == "cot")
            n_think = sum(1 for r in rows if r["meta"].get("target_format") == "think")
            n_shuf = sum(1 for r in rows if r["meta"].get("option_perm"))
            print(f"[sources] reasoning_corpus:{s['path']} fmt={s.get('target_format','letter')} "
                  f"task={s.get('reasoning_task')} only_rationale={bool(s.get('only_rationale', False))} "
                  f"shuffle_options={bool(s.get('shuffle_options', False))} "
                  f"-> {len(rows)} examples ({n_cot} cot, {n_think} think, {n_shuf} shuffled)", flush=True)
        elif s["type"] == "structural_captions":
            rows = load_structural_captions(s["path"], limit=limit)
            print(f"[sources] structural_captions:{s['path']} -> {len(rows)} examples", flush=True)
        elif s["type"] == "catsbench":
            rows = load_catsbench(split=s.get("split", "train"), limit=limit)
            print(f"[sources] catsbench:{s.get('split','train')} -> {len(rows)} examples", flush=True)
        elif s["type"] == "chatts_style_captions":
            rows = load_chatts_style_captions(s["path"], limit=limit,
                                      target_field=s.get("target_field", "caption"))
            print(f"[sources] chatts_style_captions:{s['path']} -> {len(rows)} examples", flush=True)
        elif s["type"] == "jsonl":
            rows = load_unified_jsonl(s["path"], limit=limit)
            print(f"[sources] jsonl:{s['path']} -> {len(rows)} examples", flush=True)
        else:
            raise ValueError(f"unknown source type: {s['type']}")
        # Optional per-source `task_tag` override: relabels the `task` of every
        # row so balance modes (weighted_even / weighted_even_src) treat this
        # source as its own bucket even when it shares a loader/task with another
        # source (e.g. a second `tsexam_perception` file used for reasoning QA).
        tag = s.get("task_tag")
        if tag:
            for r in rows:
                r["task"] = tag
            print(f"[sources]   -> task_tag override: '{tag}'", flush=True)
        out.extend(rows)
    print(f"[sources] total unified examples: {len(out)}", flush=True)
    return out
