"""Human spot-checks of the automated scorer (paper §3.4: spot-checks on 10% of tasks).

``sample`` draws a seeded sample of task ids, stratified by complexity, and one seed per sampled
task. It writes one CSV row per configuration that ran that (task, seed), with the gold answer,
the prediction and the automated verdict, plus empty ``human_success`` and ``notes`` columns
for a person to fill in. Grading the same task across configurations shows whether scorer errors
depend on the configuration, which would bias the comparisons.

``score`` reads the graded CSV back and reports agreement between the human and automated
verdicts: raw agreement and Cohen's kappa, overall and per configuration, each with n, plus the
rows where they disagree.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

COLUMNS = ("config", "seed", "task_id", "complexity", "question", "gold", "prediction",
           "auto_success", "auto_success_strict", "human_success", "notes")
_TRUE = {"1", "y", "yes", "true", "t", "correct"}
_FALSE = {"0", "n", "no", "false", "f", "incorrect"}


def sample(
    records: list[dict[str, Any]],
    questions: dict[str, str] | None = None,
    fraction: float = 0.1,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Rows to grade: ``round(fraction * n)`` tasks per complexity stratum (at least 1 when the
    stratum is non-empty), one seed per sampled task, every config that ran that pair."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    rng = random.Random(seed)
    cx_of: dict[str, str] = {}
    seeds_of: dict[str, set[int]] = defaultdict(set)
    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in records:
        cx_of[r["task_id"]] = r["complexity"]
        seeds_of[r["task_id"]].add(r["seed"])
        by_key[(r["task_id"], r["seed"])].append(r)

    picked: list[tuple[str, int]] = []
    for cx in sorted(set(cx_of.values())):
        ids = sorted(t for t, c in cx_of.items() if c == cx)
        k = max(1, round(fraction * len(ids)))
        for t in sorted(rng.sample(ids, k)):
            picked.append((t, rng.choice(sorted(seeds_of[t]))))

    rows = []
    for t, s in picked:
        for r in sorted(by_key[(t, s)], key=lambda r: r["config"]):
            rows.append({
                "config": r["config"], "seed": s, "task_id": t, "complexity": cx_of[t],
                "question": (questions or {}).get(t, ""), "gold": r["gold"],
                "prediction": r["prediction"], "auto_success": int(bool(r["success"])),
                "auto_success_strict": (
                    int(bool(r["success_strict"])) if "success_strict" in r else ""
                ),
                "human_success": "", "notes": "",
            })
    return rows


def write_csv(rows: list[dict[str, Any]], path: str | Path, force: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; it may hold human grades. Use --force to replace.")
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def _verdict(value: str) -> bool | None:
    v = (value or "").strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None


def _agreement(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    n = len(pairs)
    if not n:
        return {"n": 0, "agreement": None, "cohen_kappa": None}
    agree = sum(a == h for a, h in pairs) / n
    p_auto = sum(a for a, _ in pairs) / n
    p_human = sum(h for _, h in pairs) / n
    expected = p_auto * p_human + (1 - p_auto) * (1 - p_human)
    kappa = None if expected == 1 else (agree - expected) / (1 - expected)
    return {"n": n, "agreement": agree, "cohen_kappa": kappa}


def score(path: str | Path) -> dict[str, Any]:
    """Agreement between ``auto_success`` and the filled-in ``human_success`` column."""
    with Path(path).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    graded: list[tuple[dict, bool, bool]] = []
    strict_pairs: list[tuple[bool, bool]] = []
    ungraded = 0
    for r in rows:
        human = _verdict(r.get("human_success", ""))
        if human is None:
            ungraded += 1
            continue
        graded.append((r, r["auto_success"].strip() == "1", human))
        if (r.get("auto_success_strict") or "").strip() in {"0", "1"}:
            strict_pairs.append((r["auto_success_strict"].strip() == "1", human))
    by_config: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for r, a, h in graded:
        by_config[r["config"]].append((a, h))
    return {
        "rows": len(rows),
        "ungraded": ungraded,
        "overall": _agreement([(a, h) for _, a, h in graded]),
        "overall_strict": _agreement(strict_pairs),
        "by_config": {c: _agreement(p) for c, p in sorted(by_config.items())},
        "disagreements": [
            {"config": r["config"], "seed": int(r["seed"]), "task_id": r["task_id"],
             "auto_success": int(a), "human_success": int(h)}
            for r, a, h in graded if a != h
        ],
    }


def load_records(out_dir: str | Path) -> list[dict[str, Any]]:
    lines = (Path(out_dir) / "runs.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def format_score(result: dict[str, Any]) -> str:
    def fmt(a: dict[str, Any]) -> str:
        if not a["n"]:
            return "n=0"
        kappa = "–" if a["cohen_kappa"] is None else f"{a['cohen_kappa']:.2f}"
        return f"agreement {100 * a['agreement']:.1f}% (n={a['n']}), Cohen's kappa {kappa}"

    lines = [f"graded {result['rows'] - result['ungraded']} of {result['rows']} rows",
             f"overall: {fmt(result['overall'])}",
             f"strict scorer: {fmt(result.get('overall_strict', {'n': 0}))}"]
    lines += [f"  {c}: {fmt(a)}" for c, a in result["by_config"].items()]
    lines.append(f"disagreements: {len(result['disagreements'])}")
    lines += [f"  {d['config']} seed {d['seed']} {d['task_id']}: auto {d['auto_success']}, "
              f"human {d['human_success']}" for d in result["disagreements"]]
    return "\n".join(lines)
