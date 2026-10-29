"""Confidence-based early stopping for agent reasoning loops.

confidence = w * self_reported + (1 - w) * consistency(previous draft, current draft)

The loop stops once confidence exceeds the threshold (0.88 in the paper). Consistency needs two
drafts, so the earliest possible stop is after the second draft.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .config import EarlyStoppingConfig

_CONF = re.compile(r"CONFIDENCE:\s*([01](?:\.\d+)?)", re.IGNORECASE)
_ANSWER = re.compile(r"ANSWER:\s*(.+?)(?:\n[A-Z]+:|\Z)", re.IGNORECASE | re.DOTALL)


def parse_confidence(text: str) -> float | None:
    m = _CONF.search(text)
    return min(1.0, max(0.0, float(m.group(1)))) if m else None


def extract_answer(text: str) -> str:
    m = _ANSWER.search(text)
    return (m.group(1) if m else text).strip()


def consistency(prev: str, cur: str) -> float:
    a, b = extract_answer(prev).lower(), extract_answer(cur).lower()
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


class EarlyStopper:
    def __init__(self, cfg: EarlyStoppingConfig) -> None:
        self.cfg = cfg

    def confidence(self, drafts: list[str]) -> float | None:
        if len(drafts) < 2:
            return None
        self_rep = parse_confidence(drafts[-1])
        if self_rep is None:
            self_rep = 0.0
        w = self.cfg.self_report_weight
        return w * self_rep + (1 - w) * consistency(drafts[-2], drafts[-1])

    def should_stop(self, drafts: list[str]) -> tuple[bool, float | None]:
        conf = self.confidence(drafts)
        if not self.cfg.enabled or conf is None:
            return False, conf
        return conf > self.cfg.threshold, conf
