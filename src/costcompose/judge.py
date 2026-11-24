"""Optional LLM-judge audit of the deterministic scorer.

The primary and strict scores in ``runs.jsonl`` are deterministic and are the study's metrics.
This module only checks them: it asks an independent model whether each selected answer means
the same as the reference answer, and reports how often the judge agrees with each score.

* The judge must not be a model on the routing ladder (or the fixed default model): a model
  grading its own family's answers tends to favour its own style, which is the bias the audit is
  meant to detect. ``check_off_ladder`` refuses such a model.
* By default it judges only the runs where the primary and strict scores disagree, the cases
  the unit rule decides. Spot-check rows can be added so the judge can also be compared with the
  human grades.
* Nothing here writes to ``runs.jsonl`` or changes any reported success rate.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import RoutingConfig
from .llm import LLMClient
from .spotcheck import _agreement

JUDGE_PROMPT = """[role=judge]
You grade a model's answer to a question against the reference answer. Decide whether the \
model's answer states the same answer as the reference. Ignore formatting only: units or \
currency the question itself asks for, currency symbols, thousands separators, capitalisation \
and trailing punctuation. Count as INCORRECT: a different number or value, a different unit \
than the question asks for, a missing or extra list item, a hedge such as "approximately", or \
more than one candidate answer. Reply with exactly one line: VERDICT: CORRECT or VERDICT: \
INCORRECT"""

_VERDICT = re.compile(r"VERDICT:\s*(CORRECT|INCORRECT)", re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _tail(model: str) -> str:
    return model.rstrip("/").split("/")[-1].lower()


def check_off_ladder(model: str, routing: RoutingConfig) -> None:
    """Raise if ``model`` is the same model as any ladder rung or the fixed default model."""
    used = [r.model for r in routing.ladder] + [routing.default_model]
    clash = [m for m in used if _tail(m) == _tail(model)]
    if clash:
        raise ValueError(
            f"judge model {model!r} is on the routing ladder ({clash[0]}); use a model from a "
            "family the study does not route to"
        )


def parse_verdict(text: str) -> bool | None:
    found = _VERDICT.findall(_THINK.sub("", text or ""))
    return None if not found else found[-1].upper() == "CORRECT"


def select(
    records: list[dict[str, Any]], spotcheck_csv: str | Path | None = None
) -> list[dict[str, Any]]:
    """Runs where primary and strict disagree, plus any runs listed in a spot-check CSV."""
    keys: set[tuple[str, int, str]] = set()
    if spotcheck_csv:
        with Path(spotcheck_csv).open(newline="") as fh:
            keys = {(r["config"], int(r["seed"]), r["task_id"]) for r in csv.DictReader(fh)}
    out = []
    for r in records:
        disputed = "success_strict" in r and bool(r["success"]) != bool(r["success_strict"])
        if disputed or (r["config"], r["seed"], r["task_id"]) in keys:
            out.append(r)
    return out


def run(
    rows: list[dict[str, Any]],
    questions: dict[str, str],
    client: LLMClient,
    model: str,
) -> list[dict[str, Any]]:
    judged = []
    for r in rows:
        user = (f"Question: {questions.get(r['task_id'], '')}\n"
                f"Reference answer: {r['gold']}\nModel answer: {r['prediction']}")
        resp = client.complete(
            model,
            [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
            temperature=0.0,
            seed=0,
        )
        judged.append({
            "config": r["config"], "seed": r["seed"], "task_id": r["task_id"],
            "gold": r["gold"], "prediction": r["prediction"],
            "success": bool(r["success"]),
            "success_strict": bool(r.get("success_strict", r["success"])),
            "judge_correct": parse_verdict(resp.content),
            "judge_model": model,
            "judge_tokens": resp.usage.total,
        })
    return judged


def agreement(judged: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [j for j in judged if j["judge_correct"] is not None]
    by_config: dict[str, list[dict]] = defaultdict(list)
    for j in usable:
        by_config[j["config"]].append(j)

    def vs(rows: list[dict], key: str) -> dict[str, Any]:
        return _agreement([(bool(j[key]), bool(j["judge_correct"])) for j in rows])

    return {
        "judged": len(judged),
        "unparsed_verdicts": len(judged) - len(usable),
        "judge_model": judged[0]["judge_model"] if judged else None,
        "judge_vs_primary": vs(usable, "success"),
        "judge_vs_strict": vs(usable, "success_strict"),
        "by_config": {
            c: {"judge_vs_primary": vs(rs, "success"), "judge_vs_strict": vs(rs, "success_strict")}
            for c, rs in sorted(by_config.items())
        },
    }


def write(judged: list[dict[str, Any]], out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    (out / "judge.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judged))
    result = agreement(judged)
    (out / "judge_agreement.json").write_text(json.dumps(result, indent=2))
    return result


def format_result(result: dict[str, Any]) -> str:
    def fmt(a: dict[str, Any]) -> str:
        if not a["n"]:
            return "n=0"
        kappa = "–" if a["cohen_kappa"] is None else f"{a['cohen_kappa']:.2f}"
        return f"agreement {100 * a['agreement']:.1f}% (n={a['n']}), Cohen's kappa {kappa}"

    lines = [f"judge {result['judge_model']}: {result['judged']} runs judged, "
             f"{result['unparsed_verdicts']} unparsed",
             f"vs primary score: {fmt(result['judge_vs_primary'])}",
             f"vs strict score:  {fmt(result['judge_vs_strict'])}"]
    for c, a in result["by_config"].items():
        lines.append(f"  {c}: primary {fmt(a['judge_vs_primary'])}; "
                     f"strict {fmt(a['judge_vs_strict'])}")
    return "\n".join(lines)
