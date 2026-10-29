"""Aggregate per-task records into per-configuration results and hypothesis checks.

Statistics: metrics are computed per (config, seed), then reported as mean and standard
deviation across seeds, matching the paper's protocol.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np

SCALARS = (
    "cost_usd",
    "cost_reference_usd",
    "latency_p50_s",
    "latency_p95_s",
    "success_rate",
    "cache_hit_rate",
    "cross_model_hit_share",
    "llm_calls_per_task",
    "early_stops_per_task",
)


def _per_seed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lat = np.array([r["latency_s"] for r in rows])
    requests = sum(r["requests"] for r in rows)
    hits = sum(r["cache_hits"] for r in rows)
    agent_req: Counter = Counter()
    agent_hit: Counter = Counter()
    routing: Counter = Counter()
    for r in rows:
        agent_req.update(r["requests_by_agent"])
        agent_hit.update(r["cache_hits_by_agent"])
        routing.update(r["routing"])
    by_cx: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_cx[r["complexity"]].append(r["success"])
    total_routed = sum(routing.values()) or 1
    return {
        "n_tasks": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "cost_usd": float(np.mean([r["cost_usd"] for r in rows])),
        "cost_reference_usd": float(np.mean([r["cost_reference_usd"] for r in rows])),
        "latency_p50_s": float(np.percentile(lat, 50)),
        "latency_p95_s": float(np.percentile(lat, 95)),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "cache_hit_rate": hits / requests if requests else 0.0,
        "cross_model_hit_share": (
            sum(r["cross_model_hits"] for r in rows) / hits if hits else 0.0
        ),
        "llm_calls_per_task": float(np.mean([r["llm_calls"] for r in rows])),
        "early_stops_per_task": float(np.mean([r["early_stops"] for r in rows])),
        "cache_hit_rate_by_agent": {a: agent_hit[a] / n for a, n in agent_req.items() if n},
        "routing_share": {m: c / total_routed for m, c in sorted(routing.items())},
        "success_by_complexity": {k: float(np.mean(v)) for k, v in sorted(by_cx.items())},
    }


def _learning_curve(rows: list[dict[str, Any]], bins: int = 4) -> list[dict[str, float]]:
    """Success and cost per task over successive quarters of the task stream (one seed)."""
    rows = sorted(rows, key=lambda r: r.get("position", 0))
    chunks = [c for c in np.array_split(np.arange(len(rows)), bins) if len(c)]
    return [
        {
            "success_rate": float(np.mean([rows[i]["success"] for i in c])),
            "cost_usd": float(np.mean([rows[i]["cost_usd"] for i in c])),
        }
        for c in chunks
    ]


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


def _merge_dicts(dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({k for d in dicts for k in d})
    return {k: _mean_std([d.get(k, 0.0) for d in dicts]) for k in keys}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    techniques: dict[str, dict] = {}
    policies: dict[str, str] = {}
    for r in records:
        grouped[r["config"]][r["seed"]].append(r)
        techniques[r["config"]] = r.get("techniques", {})
        policies[r["config"]] = r.get("routing_policy", "heuristic")

    configs: dict[str, Any] = {}
    for name, seeds in grouped.items():
        per_seed = [_per_seed(rows) for _, rows in sorted(seeds.items())]
        curves = [_learning_curve(rows) for _, rows in sorted(seeds.items())]
        n_bins = min(len(c) for c in curves)
        configs[name] = {
            "techniques": techniques[name],
            "routing_policy": policies[name],
            "learning_curve": [
                {k: _mean_std([c[i][k] for c in curves]) for k in ("success_rate", "cost_usd")}
                for i in range(n_bins)
            ],
            "seeds": sorted(seeds),
            "n_tasks": per_seed[0]["n_tasks"],
            "errors": sum(p["errors"] for p in per_seed),
            **{k: _mean_std([p[k] for p in per_seed]) for k in SCALARS},
            **{
                k: _merge_dicts([p[k] for p in per_seed])
                for k in ("cache_hit_rate_by_agent", "routing_share", "success_by_complexity")
            },
        }
    return {"configs": configs, "hypotheses": hypotheses(configs)}


def _saving(configs: dict[str, Any], name: str) -> float | None:
    if "baseline" not in configs or name not in configs:
        return None
    base = configs["baseline"]["cost_usd"]["mean"]
    return 1.0 - configs[name]["cost_usd"]["mean"] / base if base else None


def hypotheses(configs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    s_cache, s_route = _saving(configs, "cache_only"), _saving(configs, "route_only")
    # H1: prefer the clean cache+route ablation; fall back to the full composed config.
    target = "cache_route" if "cache_route" in configs else "composed"
    s_obs = _saving(configs, target)
    if None not in (s_cache, s_route, s_obs):
        additive = s_cache + s_route
        independent = 1 - (1 - s_cache) * (1 - s_route)
        out["H1_sub_additive"] = {
            "compared_config": target,
            "observed_saving": s_obs,
            "additive_prediction": additive,
            "independent_prediction": independent,
            "observed_over_additive": s_obs / additive if additive else None,
            "observed_over_independent": s_obs / independent if independent else None,
            "note": None if target == "cache_route" else (
                "composed also enables early stopping; add configs/ablations/cache_route.yaml "
                "for a clean cache x routing comparison"
            ),
        }
    if "cache_only" in configs and "cache_only_pipeline" in configs:
        out["H2_pipeline_vs_agent_cache"] = {
            "agent_scope_hit_rate": configs["cache_only"]["cache_hit_rate"]["mean"],
            "pipeline_scope_hit_rate": configs["cache_only_pipeline"]["cache_hit_rate"]["mean"],
        }
    pairs = [("learned_route", "route_only"), ("composed_learned", "composed")]
    learned = {}
    for lrn, heur in pairs:
        if lrn in configs and heur in configs:
            learned[f"{lrn}_vs_{heur}"] = {
                "cost_ratio": configs[lrn]["cost_usd"]["mean"] / configs[heur]["cost_usd"]["mean"]
                if configs[heur]["cost_usd"]["mean"] else None,
                "success_delta": configs[lrn]["success_rate"]["mean"]
                - configs[heur]["success_rate"]["mean"],
            }
    if learned:
        out["learned_vs_heuristic_routing"] = learned
    if "baseline" in configs and "route_only" in configs:
        base = configs["baseline"]["success_by_complexity"]
        routed = configs["route_only"]["success_by_complexity"]
        out["H3_routing_success_drop"] = {
            cx: base[cx]["mean"] - routed[cx]["mean"] for cx in base if cx in routed
        }
    return out


def _fmt(ms: dict[str, float], pct: bool = False, money: bool = False) -> str:
    m, s = ms["mean"], ms["std"]
    if pct:
        return f"{100 * m:.1f} ± {100 * s:.1f}%"
    if money:
        return f"${m:.4f} ± {s:.4f}"
    return f"{m:.2f} ± {s:.2f}"


def to_markdown(summary: dict[str, Any]) -> str:
    cfgs = summary["configs"]
    order = [c for c in ("baseline", "cache_only", "route_only", "composed") if c in cfgs]
    order += sorted(c for c in cfgs if c not in order)
    lines = [
        "# Results",
        "",
        "Mean ± std across seeds. Cost is USD per task at each model's own price.",
        "",
        "| Config | Cost / task | P50 latency (s) | P95 latency (s) | Success | Cache hit "
        "| LLM calls / task |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in order:
        r = cfgs[c]
        lines.append(
            f"| {c} | {_fmt(r['cost_usd'], money=True)} | {_fmt(r['latency_p50_s'])} "
            f"| {_fmt(r['latency_p95_s'])} | {_fmt(r['success_rate'], pct=True)} "
            f"| {_fmt(r['cache_hit_rate'], pct=True)} | {_fmt(r['llm_calls_per_task'])} |"
        )
    lines += ["", "## Success by complexity", "", "| Config | low | medium | high |",
              "| --- | --- | --- | --- |"]
    for c in order:
        sbc = cfgs[c]["success_by_complexity"]
        cells = [_fmt(sbc[k], pct=True) if k in sbc else "–" for k in ("low", "medium", "high")]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "## Routing share", ""]
    for c in order:
        routing = cfgs[c]["routing_share"].items()
        share = ", ".join(f"{m}: {100 * v['mean']:.0f}%" for m, v in routing)
        lines.append(f"- {c}: {share}")
    learners = [c for c in order if cfgs[c].get("routing_policy") == "linucb"]
    if learners:
        lines += ["", "## Learned router: learning curve", "",
                  "Task stream split into quarters (per seed, then averaged).", "",
                  "| Config | Quarter | Success | Cost / task |", "| --- | --- | --- | --- |"]
        for c in learners:
            for i, q in enumerate(cfgs[c]["learning_curve"], 1):
                lines.append(f"| {c} | {i} | {_fmt(q['success_rate'], pct=True)} "
                             f"| {_fmt(q['cost_usd'], money=True)} |")
    lines += ["", "## Hypothesis checks", "", "```json"]
    lines += [json.dumps(summary["hypotheses"], indent=2), "```", ""]
    return "\n".join(lines)
