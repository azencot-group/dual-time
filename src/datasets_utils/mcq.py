"""Multiple-choice helpers shared by the loaders: gold-answer -> letter
resolution, the deterministic option shuffle and the rationale "Answer: X"
trailer remap that keeps a shuffled row consistent.
"""

from __future__ import annotations

import re

from .prompts import LETTERS


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def resolve_letter(answer: str, options: list) -> str:
    """Match `answer` against `options` and return A/B/C/...; raise on miss."""
    if not options:
        raise ValueError("empty options")
    for i, o in enumerate(options):          # exact match
        if str(o) == str(answer):
            return LETTERS[i]
    na = _norm(answer)                        # whitespace/case-normalised
    for i, o in enumerate(options):
        if _norm(o) == na:
            return LETTERS[i]
    for i, o in enumerate(options):          # substring fallback ("Option A: foo")
        if na and na in _norm(o):
            return LETTERS[i]
        if _norm(o) and _norm(o) in na:
            return LETTERS[i]
    raise ValueError(f"could not resolve answer={answer!r} in options={options!r}")


# Rationales that cite options by letter would break if the letters move.
# Any row matching this is left unshuffled (0 rows in the current TR subset).
_LETTER_REF_RE = re.compile(
    r"(?i)\boption\s+\(?[A-E]\)?\b|\banswer\s+is\s+\(?[A-E]\)?\b|\(\s*[A-E]\s*\)")


# NR/EP (and ER/TSF) rationales in the reasoning set end with an explicit
# "Answer: X." trailer that _LETTER_REF_RE does not catch. When the options are
# shuffled the gold letter moves, so the trailer must be rewritten to the new
# letter or the rationale contradicts the answer. Option text in parentheses
# after the letter is left untouched.
_ANSWER_TRAILER_RE = re.compile(r"(?i)(\banswer\s*[:=]\s*\(?)([A-E])\b")


def _remap_answer_trailer(rationale: str, old_gold: str, new_gold: str) -> str:
    if not rationale or old_gold == new_gold:
        return rationale
    return _ANSWER_TRAILER_RE.sub(
        lambda m: m.group(1) + (new_gold if m.group(2).upper() == old_gold else m.group(2)),
        rationale)


def _shuffle_mcq(opt_texts, gold_letter, rng):
    """Permute option texts with `rng`, remapping the gold letter to follow
    its text. Returns (new_texts, new_gold_letter, perm) where perm[j] is the
    ORIGINAL position of the option now shown at position j."""
    perm = list(range(len(opt_texts)))
    rng.shuffle(perm)
    new_texts = [opt_texts[j] for j in perm]
    new_gold = LETTERS[perm.index(LETTERS.index(gold_letter))]
    return new_texts, new_gold, perm
