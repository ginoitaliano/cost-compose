"""GAIA-style quasi-exact-match scoring."""

from __future__ import annotations

import re
import string

_FINAL = re.compile(r"FINAL ANSWER:\s*(.+)", re.IGNORECASE | re.DOTALL)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


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


def _match_one(pred: str, gold: str) -> bool:
    g = _as_number(gold)
    if g is not None:
        p = _as_number(pred)
        return p is not None and abs(p - g) <= 1e-6 * max(1.0, abs(g))
    return _norm_text(pred) == _norm_text(gold)


def score(prediction: str, gold: str) -> bool:
    pred = extract_final_answer(prediction)
    if _as_number(gold) is not None:
        return _match_one(pred, gold)
    if any(sep in gold for sep in (",", ";")):
        gs = [x.strip() for x in re.split(r"[,;]", gold)]
        ps = [x.strip() for x in re.split(r"[,;]", pred)]
        return len(gs) == len(ps) and all(_match_one(p, g) for p, g in zip(ps, gs, strict=True))
    return _match_one(pred, gold)
