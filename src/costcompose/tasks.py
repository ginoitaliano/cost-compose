"""Task loading. Tasks are JSONL rows: id, question, answer, complexity, source."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

COMPLEXITIES = ("low", "medium", "high")


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    answer: str
    complexity: str
    source: str = "custom"


def load_tasks(path: str | Path, limit: int | None = None) -> list[Task]:
    tasks = []
    with Path(path).open() as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("complexity") not in COMPLEXITIES:
                raise ValueError(f"{path}:{n}: complexity must be one of {COMPLEXITIES}")
            tasks.append(
                Task(
                    id=str(row["id"]),
                    question=row["question"],
                    answer=str(row["answer"]),
                    complexity=row["complexity"],
                    source=row.get("source", "custom"),
                )
            )
    ids = [t.id for t in tasks]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        raise ValueError(f"duplicate task ids: {dupes[:5]}")
    return tasks[:limit] if limit else tasks


def ordered_for_seed(tasks: list[Task], seed: int) -> list[Task]:
    """Task order is shuffled per seed. Order matters: it decides what the cache has seen."""
    out = list(tasks)
    random.Random(seed).shuffle(out)
    return out


def strata(tasks: list[Task]) -> dict[str, int]:
    return dict(Counter(t.complexity for t in tasks))
