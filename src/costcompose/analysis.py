"""Aggregate per-task records into per-configuration results and hypothesis checks.

Statistics: metrics are computed per (config, seed), then reported as mean and standard
deviation across seeds, matching the paper's protocol. Comparisons between configurations use
paired resampling over tasks: every configuration runs the same tasks, so the unit of
resampling is the task (with all of its seeds), not the seed.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np

N_BOOT = 10_000  # bootstrap resamples
BOOT_SEED = 0  # fixed so the report is reproducible from runs.jsonl


def _task_means(records: list[dict[str, Any]], config: str, key: str) -> dict[str, float]:
    """Per-task value of ``key`` for one config, averaged over that task's seeds."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r["config"] == config:
            by_task[r["task_id"]].append(float(r[key]))
    return {t: float(np.mean(v)) for t, v in by_task.items()}


def _paired_matrix(
    records: list[dict[str, Any]], configs: list[str], key: str
) -> tuple[list[str], np.ndarray]:
    """Tasks present in every config, and a [len(configs), n_tasks] matrix of task means."""
    means = [_task_means(records, c, key) for c in configs]
    common = sorted(set.intersection(*(set(m) for m in means))) if means else []
    return common, np.array([[m[t] for t in common] for m in means], dtype=float)


def _boot_index(n: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _ci95(samples: np.ndarray) -> list[float]:
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return [float(lo), float(hi)]


def _h1_test(records: list[dict[str, Any]], target: str) -> dict[str, Any] | None:
    """H1 against the independence baseline, with a paired bootstrap CI over tasks.

    Savings are ratios of total cost over the same tasks: s = 1 - cost(config) / cost(baseline).
    If caching and routing act independently, the composed saving is 1 - (1-s_c)(1-s_r).
    The interaction term is observed minus that prediction.
    """
    names = ["baseline", "cache_only", "route_only", target]
    tasks, m = _paired_matrix(records, names, "cost_usd")
    if not tasks:
        return None

    def stats(mat: np.ndarray) -> tuple[np.ndarray, ...]:
        totals = mat.sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_cache, s_route, s_obs = (1.0 - totals[i] / totals[0] for i in (1, 2, 3))
        independent = 1.0 - (1.0 - s_cache) * (1.0 - s_route)
        return s_cache, s_route, s_obs, independent, s_cache + s_route

    s_cache, s_route, s_obs, independent, additive = (float(x) for x in stats(m))
    b_cache, b_route, b_obs, b_ind, _ = stats(m[:, _boot_index(len(tasks))])
    interaction = s_obs - independent
    ci = _ci95(b_obs - b_ind)
    if ci[1] < 0:
        verdict = "sub-independent: CI below 0 (techniques interfere)"
    elif ci[0] > 0:
        verdict = "super-independent: CI above 0 (techniques reinforce)"
    else:
        verdict = "consistent with independence: CI contains 0"
    return {
        "compared_config": target,
        "n_tasks": len(tasks),
        "saving_cache_only": s_cache,
        "saving_route_only": s_route,
        "observed_saving": s_obs,
        "observed_saving_ci95": _ci95(b_obs),
        "primary_independence_test": {
            "independent_prediction": independent,
            "interaction": interaction,
            "interaction_ci95": ci,
            "observed_over_independent": s_obs / independent if independent else None,
            "verdict": verdict,
        },
        "secondary_sum_of_savings": {
            "additive_prediction": additive,
            "observed_over_additive": s_obs / additive if additive else None,
            "note": "secondary: any two positive savings compose below their sum "
                    "(a + b - ab < a + b), so falling short of the sum is expected "
                    "without interference",
        },
        "note": None if target == "cache_route" else (
            "composed also enables early stopping; add configs/ablations/cache_route.yaml "
            "for a clean cache x routing comparison"
        ),
    }

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


def _rate(k: float, n: float) -> dict[str, Any]:
    """A rate with its denominator, so every reported rate carries its sample size."""
    return {"rate": float(k / n) if n else None, "n": int(n)}


def _cache_correctness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled over all seeds of one config. Lookup-level and task-level views.

    Lookup level: of the cache lookups that hit (missed), the share whose task succeeded.
    Task level: success of tasks that had at least one hit vs tasks that had none.
    """
    lookups = sum(r["requests"] for r in rows)
    hits = sum(r["cache_hits"] for r in rows)
    misses = lookups - hits
    known = all("entity_mismatch_hits" in r for r in rows)
    mism = sum(r.get("entity_mismatch_hits", 0) for r in rows)
    with_hit = [r for r in rows if r["cache_hits"] > 0]
    without = [r for r in rows if r["cache_hits"] == 0]
    by_agent: Counter = Counter()
    for r in rows:
        by_agent.update(r.get("entity_mismatch_hits_by_agent", {}))
    return {
        "scope": next((r.get("cache_scope") for r in rows if r.get("cache_scope")), None),
        "hit_rate": _rate(hits, lookups),
        "entity_mismatch_share_of_hits": _rate(mism, hits) if known else None,
        "entity_mismatch_hits_by_agent": dict(by_agent) if known else None,
        "success_given_hit": _rate(sum(r["cache_hits"] * r["success"] for r in rows), hits),
        "success_given_miss": _rate(
            sum((r["requests"] - r["cache_hits"]) * r["success"] for r in rows), misses
        ),
        "success_given_entity_mismatch_hit": (
            _rate(sum(r.get("entity_mismatch_hits", 0) * r["success"] for r in rows), mism)
            if known else None
        ),
        "task_success_with_any_hit": _rate(sum(r["success"] for r in with_hit), len(with_hit)),
        "task_success_with_no_hit": _rate(sum(r["success"] for r in without), len(without)),
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
            "cache_correctness": (
                _cache_correctness([r for rows in seeds.values() for r in rows])
                if techniques[name].get("caching") else None
            ),
            "n_tasks": per_seed[0]["n_tasks"],
            "errors": sum(p["errors"] for p in per_seed),
            **{k: _mean_std([p[k] for p in per_seed]) for k in SCALARS},
            **{
                k: _merge_dicts([p[k] for p in per_seed])
                for k in ("cache_hit_rate_by_agent", "routing_share", "success_by_complexity")
            },
        }
    return {"configs": configs, "hypotheses": hypotheses(configs, records)}


def hypotheses(configs: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # H1: prefer the clean cache+route ablation; fall back to the full composed config.
    target = "cache_route" if "cache_route" in configs else "composed"
    if all(c in configs for c in ("baseline", "cache_only", "route_only", target)):
        h1 = _h1_test(records, target)
        if h1 is not None:
            out["H1_sub_additive"] = h1
    if "cache_only" in configs and "cache_only_pipeline" in configs:
        agent_cc = configs["cache_only"]["cache_correctness"] or {}
        pipe_cc = configs["cache_only_pipeline"]["cache_correctness"] or {}
        out["H2_pipeline_vs_agent_cache"] = {
            "agent_scope_hit_rate": configs["cache_only"]["cache_hit_rate"]["mean"],
            "pipeline_scope_hit_rate": configs["cache_only_pipeline"]["cache_hit_rate"]["mean"],
            "agent_scope_entity_mismatch_share": agent_cc.get("entity_mismatch_share_of_hits"),
            "pipeline_scope_entity_mismatch_share": pipe_cc.get(
                "entity_mismatch_share_of_hits"
            ),
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


def _pct_ci(ci: list[float]) -> str:
    return f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def _r(x: dict[str, Any] | None) -> str:
    """Format a {rate, n} pair as 'xx.x% (n=N)'."""
    if not x:
        return "–"
    if x["rate"] is None:
        return f"– (n={x['n']})"
    return f"{100 * x['rate']:.1f}% (n={x['n']})"


def _cache_markdown(cfgs: dict[str, Any], order: list[str]) -> list[str]:
    rows = [c for c in order if cfgs[c].get("cache_correctness")]
    if not rows:
        return []
    lines = [
        "", "## Cache correctness", "",
        "Pooled over seeds. n is cache lookups for the hit rate, hits for the mismatch share and "
        "success given a hit, misses for success given a miss, and tasks for the task-level "
        "columns. An entity-mismatch hit served an entry stored for a request with different "
        "IDs, numbers or dates.", "",
        "| Config | Scope | Hit rate | Entity-mismatch share of hits | Success given hit "
        "| Success given miss | Task success, any hit | Task success, no hit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in rows:
        cc = cfgs[c]["cache_correctness"]
        lines.append(
            f"| {c} | {cc['scope']} | {_r(cc['hit_rate'])} "
            f"| {_r(cc['entity_mismatch_share_of_hits'])} | {_r(cc['success_given_hit'])} "
            f"| {_r(cc['success_given_miss'])} | {_r(cc['task_success_with_any_hit'])} "
            f"| {_r(cc['task_success_with_no_hit'])} |"
        )
    return lines


def _h1_markdown(h1: dict[str, Any] | None) -> list[str]:
    if not h1:
        return []
    p, s = h1["primary_independence_test"], h1["secondary_sum_of_savings"]
    lines = [
        "", "## H1: composition of caching and routing", "",
        f"Compared config: `{h1['compared_config']}`, n = {h1['n_tasks']} tasks "
        f"(paired, all seeds per task). Savings are relative to `baseline` cost.", "",
        "| Quantity | Value |", "| --- | --- |",
        f"| Saving, cache_only | {100 * h1['saving_cache_only']:.1f}% |",
        f"| Saving, route_only | {100 * h1['saving_route_only']:.1f}% |",
        f"| Saving, {h1['compared_config']} (observed) | {100 * h1['observed_saving']:.1f}% "
        f"95% CI {_pct_ci(h1['observed_saving_ci95'])} |",
        f"| Primary: independence prediction 1 - (1 - s_cache)(1 - s_route) "
        f"| {100 * p['independent_prediction']:.1f}% |",
        f"| Primary: interaction (observed - independence) | {100 * p['interaction']:+.1f} pp "
        f"95% CI {_pct_ci(p['interaction_ci95'])} |",
        f"| Secondary: sum of savings s_cache + s_route | {100 * s['additive_prediction']:.1f}% |",
        "",
        f"Primary test: {p['verdict']}.",
        "The sum of savings is secondary: two positive savings always compose below their sum.",
    ]
    if h1.get("note"):
        lines += ["", f"Note: {h1['note']}."]
    return lines


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
    lines += _cache_markdown(cfgs, order)
    lines += _h1_markdown(summary["hypotheses"].get("H1_sub_additive"))
    lines += ["", "## Hypothesis checks", "", "```json"]
    lines += [json.dumps(summary["hypotheses"], indent=2), "```", ""]
    return "\n".join(lines)
