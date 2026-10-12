"""Prompt templates for the multiple-choice tasks (letter-only output)."""

from __future__ import annotations

from typing import Sequence

LETTERS = "ABCDEFGH"  # supports up to 8 options; the datasets use 2-4

SYSTEM = (
    "You answer multiple-choice questions about time series. "
    "Reply with ONLY the single capital letter of the correct option."
)


def format_options(options: Sequence[str]) -> str:
    return "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))


def user_text(question: str, options: Sequence[str]) -> str:
    return f"{question}\nOptions:\n{format_options(options)}\nAnswer:"


# --- TSRBench-style prompting (verbatim from the benchmark's vision inference) ---

def format_options_tsrbench(options: Sequence[str]) -> str:
    """TSRBench uses 'A. ' (period) as the option separator, not 'A) '."""
    return "".join(f"{LETTERS[i]}. {opt}\n" for i, opt in enumerate(options))


def user_text_tsrbench(question: str, options: Sequence[str],
                       series_names: Sequence[str] | None = None) -> str:
    """Verbatim TSRBench user prompt:
        [series_info]{question}\n\nSelect from the options below:{choices}\nOutput your reasoning and answer in JSON format.
    No system role. Series-names preamble only when names are available.
    """
    if series_names:
        prefix = "The time series in the image correspond to: " + ", ".join(series_names) + ".\n\n"
        question = prefix + question
    choice_text = format_options_tsrbench(options)
    return f"{question}\n\nSelect from the options below:{choice_text}\nOutput your reasoning and answer in JSON format."


def assistant_json_tsrbench(reasoning: str, gold_letter: str) -> str:
    """Format a gold assistant turn as TSRBench-style JSON."""
    import json as _json
    return _json.dumps({"reasoning": reasoning, "answer": gold_letter}, ensure_ascii=False)
