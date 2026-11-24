"""Answer scoring.

Two scores are computed for every task:

* ``score`` (strict): GAIA-style quasi-exact match. Numbers are compared numerically after
  removing ``$``, ``%`` and thousands separators; strings are compared after lower-casing and
  removing punctuation and articles; lists element-wise. Anything else after a number, such as a
  unit, makes it wrong. This matches how GAIA is usually scored.
* ``score_lenient`` (primary): the same, except that a numeric answer may be followed by unit
  words that the question itself names. ``95 USD`` counts for a question ending "in USD";
  ``1 day`` does not count for a question asking "in hours", and scale words such as ``thousand``
  never count. This keeps the success metric from depending on whether a model obeys an
  instruction to drop units, which differs by model and so would bias routing comparisons.

The rule was fixed before any study data was collected; both scores are recorded and reported.
"""

from __future__ import annotations

import re
import string

_FINAL = re.compile(r"FINAL ANSWER:\s*(.+)", re.IGNORECASE | re.DOTALL)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
# A leading number (optional currency sign, sign, thousands separators, decimals, percent),
# then whatever follows it.
_LEADING_NUMBER = re.compile(r"^\s*\$?\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*%?\s*(.*)$")
_WORD = re.compile(r"[a-z]+")
# Words that change the value or hedge it: never accepted after a number.
_SCALE_OR_HEDGE = {
    "thousand", "thousands", "million", "millions", "billion", "billions", "hundred",
    "hundreds", "k", "m", "bn", "approx", "approximately", "about", "around", "roughly",
    "estimated", "estimate", "minus", "plus", "or", "to", "between", "and", "not", "no",
}
# Connecting words allowed inside a unit phrase ("GPU-hours per week").
_UNIT_GLUE = {"per", "a", "an", "each", "the", "of", "in"}


def extract_final_answer(output: str) -> str:
    m = _FINAL.search(output or "")
    lines = (m.group(1) if m else output or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def _as_number(s: str) -> float | None:
    cleaned = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _norm_text(s: str) -> str:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(_ARTICLES.sub(" ", s).split())


def _stem(word: str) -> str:
    """Crude stem so 'hours' matches 'hour' and 'weekly' matches 'week'."""
    for suffix in ("ly", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _question_words(question: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(question.lower())}


def _numbers_equal(p: float, g: float) -> bool:
    return abs(p - g) <= 1e-6 * max(1.0, abs(g))


def _match_one(pred: str, gold: str) -> bool:
    g = _as_number(gold)
    if g is not None:
        p = _as_number(pred)
        return p is not None and _numbers_equal(p, g)
    return _norm_text(pred) == _norm_text(gold)


def _match_one_lenient(pred: str, gold: str, qwords: set[str]) -> bool:
    if _match_one(pred, gold):
        return True
    g = _as_number(gold)
    if g is None:
        return False
    m = _LEADING_NUMBER.match(pred)
    if not m or not _numbers_equal(float(m.group(1).replace(",", "")), g):
        return False
    tail = m.group(2).strip().rstrip(".")
    words = _WORD.findall(tail.lower())
    if not words or len(_WORD.sub("", tail.lower()).strip(" -/.()")) > 0:
        return False  # nothing but non-words after the number, or symbols beyond simple joiners
    if any(w in _SCALE_OR_HEDGE for w in words):
        return False
    units = [w for w in words if w not in _UNIT_GLUE]
    return bool(units) and all(_stem(w) in qwords for w in units)


def _split_list(s: str) -> list[str]:
    return [x.strip() for x in re.split(r"[,;]", s)]


def _is_list(gold: str) -> bool:
    return _as_number(gold) is None and any(sep in gold for sep in (",", ";"))


def score(prediction: str, gold: str) -> bool:
    """Strict GAIA-style score."""
    pred = extract_final_answer(prediction)
    if _is_list(gold):
        gs, ps = _split_list(gold), _split_list(pred)
        return len(gs) == len(ps) and all(_match_one(p, g) for p, g in zip(ps, gs, strict=True))
    return _match_one(pred, gold)


def score_lenient(prediction: str, gold: str, question: str) -> bool:
    """Primary score: strict, plus a trailing unit that the question names is allowed."""
    pred = extract_final_answer(prediction)
    qwords = _question_words(question)
    if _is_list(gold):
        gs, ps = _split_list(gold), _split_list(pred)
        return len(gs) == len(ps) and all(
            _match_one_lenient(p, g, qwords) for p, g in zip(ps, gs, strict=True)
        )
    return _match_one_lenient(pred, gold, qwords)
