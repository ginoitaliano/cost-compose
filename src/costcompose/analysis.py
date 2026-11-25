"""Aggregate per-task records into per-configuration results and hypothesis checks.

Headline statistics are paired comparisons over tasks. Every configuration runs the same
tasks with the same seeds, so:

* success: exact McNemar test on (task, seed) pairs, plus a paired bootstrap CI on the
  success-rate difference;
* cost and latency: paired bootstrap over tasks (each task's value averaged over its seeds),
  10,000 resamples with a fixed seed, 95% percentile CIs;
* H3: per-stratum success drop under routing with a stratified bootstrap CI, and a CI on the
  difference-of-differences (high-stratum drop minus low-stratum drop).

Per-config descriptives (mean and std across seeds, the paper's protocol) are still reported,
below the paired comparisons. Every rate is printed with its sample size.
"""

from __future__ import annotations

import json
import math
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


def _h1_test(
    records: list[dict[str, Any]], target: str, cache: str = "cache_only"
) -> dict[str, Any] | None:
    """H1 against the independence baseline, with a paired bootstrap CI over tasks.

    Savings are ratios of total cost over the same tasks: s = 1 - cost(config) / cost(baseline).
    If caching and routing act independently, the composed saving is 1 - (1-s_c)(1-s_r).
    The interaction term is observed minus that prediction.
    """
    names = ["baseline", cache, "route_only", target]
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
        "cache_config": cache,
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
            "note": "secondary: two positive savings compose below their sum under "
                    "independence (a + b - ab < a + b); with a negative saving the independence "
                    "prediction is above the sum, so the sum is not a test of interference",
        },
        "note": None if target.startswith("cache_route") else (
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
    "cache_induced_early_stops_per_task",
)


def _lookups(r: dict[str, Any]) -> int:
    """Cache lookups in a task run. Older records lack the field; every request was a lookup."""
    return r.get("cache_lookups", r["requests"])


def _per_seed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lat = np.array([r["latency_s"] for r in rows])
    requests = sum(_lookups(r) for r in rows)
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
        "cache_induced_early_stops_per_task": float(
            np.mean([r.get("cache_induced_early_stops", 0) for r in rows])
        ),
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
    lookups = sum(_lookups(r) for r in rows)
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
            sum((_lookups(r) - r["cache_hits"]) * r["success"] for r in rows), misses
        ),
        "success_given_entity_mismatch_hit": (
            _rate(sum(r.get("entity_mismatch_hits", 0) * r["success"] for r in rows), mism)
            if known else None
        ),
        "task_success_with_any_hit": _rate(sum(r["success"] for r in with_hit), len(with_hit)),
        "task_success_with_no_hit": _rate(sum(r["success"] for r in without), len(without)),
        "mechanisms": _hit_mechanisms(rows, hits, lookups),
        "subtask_cache": _subtask_cache(rows),
    }


def _subtask_cache(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Resolved-subtask cache: lookups, hits by origin, and task success with vs without a hit."""
    if not any(r.get("subtask_cache_lookups") for r in rows):
        return None
    lookups = sum(r["subtask_cache_lookups"] for r in rows)
    hits = sum(r["subtask_cache_hits"] for r in rows)
    by_origin: Counter = Counter()
    for r in rows:
        by_origin.update(r.get("subtask_hits_by_origin", {}))
    with_hit = [r for r in rows if r["subtask_cache_hits"] > 0]
    without = [r for r in rows if r["subtask_cache_hits"] == 0]
    return {
        "hit_rate": _rate(hits, lookups),
        "share_of_hits": {o: _rate(by_origin[o], hits) for o in ORIGINS},
        "task_success_with_subtask_hit": _rate(sum(r["success"] for r in with_hit),
                                               len(with_hit)),
        "task_success_without": _rate(sum(r["success"] for r in without), len(without)),
    }


ORIGINS = ("same_loop", "same_task", "cross_task")


def _hit_mechanisms(rows: list[dict[str, Any]], hits: int, lookups: int) -> dict[str, Any] | None:
    """Where served entries came from, and what the guards refused or bypassed.

    same_loop: written earlier by the reasoning loop it is served back into.
    same_task: written elsewhere in the same task run (another agent, loop or role).
    cross_task: written while an earlier task ran: the reuse a semantic cache is meant for.
    Success given a hit of each origin weights each task's success by its hits of that origin.
    """
    if not all("hits_by_origin" in r for r in rows):
        return None  # records written before origin tracking
    by_origin: Counter = Counter()
    succ: Counter = Counter()
    by_role: Counter = Counter()
    for r in rows:
        by_origin.update(r["hits_by_origin"])
        by_role.update(r.get("hits_by_role", {}))
        for o, k in r["hits_by_origin"].items():
            succ[o] += k * r["success"]
    return {
        "share_of_hits": {o: _rate(by_origin[o], hits) for o in ORIGINS},
        "success_given_hit": {o: _rate(succ[o], by_origin[o]) for o in ORIGINS},
        "cross_role_share_of_hits": _rate(sum(r["cross_role_hits"] for r in rows), hits),
        "query_entity_mismatch_share_of_hits": _rate(
            sum(r["query_entity_mismatch_hits"] for r in rows), hits
        ),
        "entity_guard_refusals": _rate(sum(r["entity_guard_refusals"] for r in rows), lookups),
        "bypassed_requests": sum(r["cache_bypassed"] for r in rows),
        "hits_by_role": dict(sorted(by_role.items())),
    }


RUNG_CUTOFFS = (0.25, 0.5, 0.75)  # configs/base.yaml ladder: score below a cutoff stays below


def _routing_scores(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Distribution of routing scores per role, against the ladder cutoffs."""
    if not all("routing_scores" in r for r in rows):
        return None
    by_role: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for role, score in r["routing_scores"]:
            by_role[role].append(score)
    out = {}
    for role, xs in sorted(by_role.items()):
        a = np.array(xs)
        out[role] = {
            "n": len(xs),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
            "share_at_or_above": {str(c): float((a >= c).mean()) for c in RUNG_CUTOFFS},
        }
    return out


def _scoring(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Primary (lenient) vs strict success, pooled over seeds. None for older records."""
    if not rows or any("success_strict" not in r for r in rows):
        return None
    n = len(rows)
    return {
        "primary": _rate(sum(bool(r["success"]) for r in rows), n),
        "strict": _rate(sum(bool(r["success_strict"]) for r in rows), n),
        "scored_differently": sum(bool(r["success"]) != bool(r["success_strict"]) for r in rows),
    }


def _cache_early_stop(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fourth composition effect: early stops triggered by drafts replayed from the cache."""
    stops = sum(r["early_stops"] for r in rows)
    induced = sum(r.get("cache_induced_early_stops", 0) for r in rows)
    drafts = sum(r["reasoning_iterations"] for r in rows)
    cached = sum(r.get("cached_drafts", 0) for r in rows)
    with_induced = [r for r in rows if r.get("cache_induced_early_stops", 0) > 0]
    without = [r for r in rows if r.get("cache_induced_early_stops", 0) == 0]
    return {
        "cache_induced_early_stops": induced,
        "n_tasks": len(rows),
        "per_task": induced / len(rows) if rows else None,
        "share_of_early_stops": _rate(induced, stops),
        "cached_share_of_drafts": _rate(cached, drafts),
        "replay_share_of_cached_drafts": _rate(
            sum(r.get("cached_draft_replays", 0) for r in rows), cached
        ),
        "task_success_with_induced_stop": _rate(
            sum(r["success"] for r in with_induced), len(with_induced)
        ),
        "task_success_without_induced_stop": _rate(
            sum(r["success"] for r in without), len(without)
        ),
    }


def _sample_sizes(seeds: dict[int, list[dict]], techniques: dict[str, bool]) -> dict[str, Any]:
    rows = [r for rs in seeds.values() for r in rs]
    routed: Counter = Counter()
    for r in rows:
        routed.update(r["routing"])
    first = sorted(seeds)[0]
    quarters = [len(c) for c in np.array_split(np.arange(len(seeds[first])), 4) if len(c)]
    return {
        "task_runs": len(rows),  # tasks x seeds: denominator of success_rate
        "cache_lookups": sum(_lookups(r) for r in rows) if techniques.get("caching") else 0,
        "routed_requests": sum(routed.values()),
        "by_complexity": dict(sorted(Counter(r["complexity"] for r in rows).items())),
        "learning_curve_task_runs": [q * len(seeds) for q in quarters],
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
            "scoring": _scoring([r for rows in seeds.values() for r in rows]),
            "cache_early_stop": (
                _cache_early_stop([r for rows in seeds.values() for r in rows])
                if techniques[name].get("caching") and techniques[name].get("early_stopping")
                else None
            ),
            "cache_correctness": (
                _cache_correctness([r for rows in seeds.values() for r in rows])
                if techniques[name].get("caching") else None
            ),
            "routing_scores": (
                _routing_scores([r for rows in seeds.values() for r in rows])
                if techniques[name].get("routing") else None
            ),
            "n_tasks": per_seed[0]["n_tasks"],
            "n": _sample_sizes(seeds, techniques[name]),
            "errors": sum(p["errors"] for p in per_seed),
            **{k: _mean_std([p[k] for p in per_seed]) for k in SCALARS},
            **{
                k: _merge_dicts([p[k] for p in per_seed])
                for k in ("cache_hit_rate_by_agent", "routing_share", "success_by_complexity")
            },
        }
    return {
        "configs": configs,
        "comparisons": _comparisons(configs, records),
        "hypotheses": hypotheses(configs, records),
    }


# Ablation -> the config it modifies. Each is compared with its parent as well as with baseline.
PARENTS = {
    "cache_only_pipeline": "cache_only",
    "learned_route": "route_only",
    "composed_learned": "composed",
    "composed_model_aware": "composed",
    "composed_exclude_cached_drafts": "composed",
    "composed_skip_tool_calls": "composed",
    "single_agent_route": "single_agent_baseline",
    "cache_only_guarded": "cache_only",
    "composed_guarded": "composed",
    "cache_route_guarded": "cache_route",
    "cache_only_role_scoped": "cache_only",
    "cache_only_first_step": "cache_only",
    "cache_only_entity_guard": "cache_only",
    "cache_only_skip_tool_calls": "cache_only",
    "cache_only_structured": "cache_only",
    "cache_route_structured": "cache_route",
    "composed_structured": "composed",
    "cache_only_structured_keys": "cache_only",
}

# Further pairs worth a paired test: the proposed cache against the guarded one, and its parts.
EXTRA_PAIRS = [
    ("cache_only_guarded", "cache_only_structured"),
    ("cache_route_guarded", "cache_route_structured"),
    ("composed_guarded", "composed_structured"),
    ("cache_only_structured_keys", "cache_only_structured"),
]


def _mcnemar_exact(ref: list[bool], cfg: list[bool]) -> dict[str, Any]:
    """Two-sided exact McNemar test: binomial test on the discordant pairs."""
    ref_only = sum(1 for a, b in zip(ref, cfg, strict=True) if a and not b)
    cfg_only = sum(1 for a, b in zip(ref, cfg, strict=True) if b and not a)
    n = ref_only + cfg_only
    if n == 0:
        p = 1.0
    else:
        k = min(ref_only, cfg_only)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return {"n_pairs": len(ref), "reference_only_success": ref_only,
            "config_only_success": cfg_only, "p_value": float(p)}


def _paired_delta(
    records: list[dict[str, Any]], ref: str, cfg: str, key: str, relative: bool = False
) -> dict[str, Any] | None:
    tasks, m = _paired_matrix(records, [ref, cfg], key)
    if not tasks:
        return None
    idx = _boot_index(len(tasks))
    b_ref, b_cfg = m[0][idx].mean(axis=1), m[1][idx].mean(axis=1)
    out = {
        "n_tasks": len(tasks),
        "reference_mean": float(m[0].mean()),
        "config_mean": float(m[1].mean()),
        "delta": float(m[1].mean() - m[0].mean()),
        "delta_ci95": _ci95(b_cfg - b_ref),
    }
    if relative:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["relative"] = float(m[1].mean() / m[0].mean() - 1) if m[0].mean() else None
            out["relative_ci95"] = _ci95(b_cfg / b_ref - 1)
    return out


def compare(records: list[dict[str, Any]], ref: str, cfg: str) -> dict[str, Any] | None:
    """Paired comparison of ``cfg`` against ``ref`` on the tasks and seeds both ran."""
    a = {(r["task_id"], r["seed"]): bool(r["success"]) for r in records if r["config"] == ref}
    b = {(r["task_id"], r["seed"]): bool(r["success"]) for r in records if r["config"] == cfg}
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    strict = None
    rows_ab = [r for r in records if r["config"] in (ref, cfg)]
    if rows_ab and all("success_strict" in r for r in rows_ab):
        sa = {(r["task_id"], r["seed"]): bool(r["success_strict"])
              for r in rows_ab if r["config"] == ref}
        sb = {(r["task_id"], r["seed"]): bool(r["success_strict"])
              for r in rows_ab if r["config"] == cfg}
        strict = {
            **_paired_delta(records, ref, cfg, "success_strict"),
            "mcnemar": _mcnemar_exact([sa[k] for k in keys], [sb[k] for k in keys]),
        }
    return {
        "reference": ref,
        "config": cfg,
        "success": {
            **_paired_delta(records, ref, cfg, "success"),
            "mcnemar": _mcnemar_exact([a[k] for k in keys], [b[k] for k in keys]),
        },
        "success_strict": strict,
        "cost_usd": _paired_delta(records, ref, cfg, "cost_usd", relative=True),
        "latency_s": _paired_delta(records, ref, cfg, "latency_s"),
    }


def _comparisons(configs: dict[str, Any], records: list[dict[str, Any]]) -> list[dict]:
    pairs = [("baseline", c) for c in configs if c != "baseline" and "baseline" in configs]
    pairs += [(p, c) for c, p in PARENTS.items() if c in configs and p in configs]
    pairs += [(p, c) for p, c in EXTRA_PAIRS if c in configs and p in configs]
    return [x for x in (compare(records, a, b) for a, b in pairs) if x is not None]


def _h3(
    records: list[dict[str, Any]], settings: dict[str, tuple[str, str]]
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]] | None:
    """Routing success drop (no routing - routing) per complexity stratum, for each setting.

    ``settings`` maps a name (e.g. "multi_agent") to (config without routing, config with
    routing). All settings use the tasks they share, and every stratum is resampled with the
    same task indices for all settings, so differences between settings stay paired.
    """
    names = [c for pair in settings.values() for c in pair]
    tasks, m = _paired_matrix(records, names, "success")
    if not tasks:
        return None
    cx_of = {r["task_id"]: r["complexity"] for r in records}
    rng = np.random.default_rng(BOOT_SEED)
    out: dict[str, Any] = {
        k: {"baseline_config": b, "routed_config": r, "by_stratum": {}}
        for k, (b, r) in settings.items()
    }
    boots: dict[str, dict[str, np.ndarray]] = {k: {} for k in settings}
    for cx in ("low", "medium", "high"):
        sel = [i for i, t in enumerate(tasks) if cx_of[t] == cx]
        if not sel:
            continue
        idx = rng.integers(0, len(sel), size=(N_BOOT, len(sel)))
        for j, name in enumerate(settings):
            base, routed = m[2 * j][sel], m[2 * j + 1][sel]
            boots[name][cx] = base[idx].mean(axis=1) - routed[idx].mean(axis=1)
            out[name]["by_stratum"][cx] = {
                "n_tasks": len(sel),
                "baseline_success": float(base.mean()),
                "routed_success": float(routed.mean()),
                "drop": float(base.mean() - routed.mean()),
                "drop_ci95": _ci95(boots[name][cx]),
            }
    for name, res in out.items():
        by = res["by_stratum"]
        if "high" in by and "low" in by:
            res["high_minus_low"] = {
                "estimate": by["high"]["drop"] - by["low"]["drop"],
                "ci95": _ci95(boots[name]["high"] - boots[name]["low"]),
            }
    return out, boots


def _h3_drop(
    records: list[dict[str, Any]], base: str, routed: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    res = _h3(records, {"setting": (base, routed)})
    return None if res is None else (res[0]["setting"], res[1]["setting"])


def _h3_test(records: list[dict[str, Any]], configs: dict[str, Any]) -> dict[str, Any] | None:
    settings = {}
    if "baseline" in configs and "route_only" in configs:
        settings["multi_agent"] = ("baseline", "route_only")
    if "single_agent_baseline" in configs and "single_agent_route" in configs:
        settings["single_agent"] = ("single_agent_baseline", "single_agent_route")
    if not settings:
        return None
    res = _h3(records, settings)
    if res is None:
        return None
    out, boots = res
    if all("high_minus_low" in out.get(k, {}) for k in ("multi_agent", "single_agent")):
        diff = {
            k: boots[k]["high"] - boots[k]["low"] for k in ("multi_agent", "single_agent")
        }
        out["multi_minus_single"] = {
            "description": "multi-agent high-minus-low drop minus single-agent high-minus-low "
                           "drop; > 0 means routing hurts complex tasks more in the multi-agent "
                           "pipeline than in the single agent",
            "estimate": out["multi_agent"]["high_minus_low"]["estimate"]
            - out["single_agent"]["high_minus_low"]["estimate"],
            "ci95": _ci95(diff["multi_agent"] - diff["single_agent"]),
        }
    return out


def hypotheses(configs: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # H1: prefer the clean cache+route ablation; fall back to the full composed config.
    target = "cache_route" if "cache_route" in configs else "composed"
    if all(c in configs for c in ("baseline", "cache_only", "route_only", target)):
        h1 = _h1_test(records, target)
        if h1 is not None:
            out["H1_sub_additive"] = h1
    # H1 again with the guarded cache: does the interaction survive once false hits are removed?
    if all(c in configs for c in ("baseline", "cache_only_guarded", "route_only",
                                  "cache_route_guarded")):
        h1g = _h1_test(records, "cache_route_guarded", cache="cache_only_guarded")
        if h1g is not None:
            out["H1_guarded_cache"] = h1g
    if all(c in configs for c in ("baseline", "cache_only_structured", "route_only",
                                  "cache_route_structured")):
        h1s = _h1_test(records, "cache_route_structured", cache="cache_only_structured")
        if h1s is not None:
            out["H1_structured_cache"] = h1s
    if "cache_only" in configs and "cache_only_pipeline" in configs:
        agent_cc = configs["cache_only"]["cache_correctness"] or {}
        pipe_cc = configs["cache_only_pipeline"]["cache_correctness"] or {}
        out["H2_pipeline_vs_agent_cache"] = {
            "agent_scope_hit_rate": configs["cache_only"]["cache_hit_rate"]["mean"],
            "pipeline_scope_hit_rate": configs["cache_only_pipeline"]["cache_hit_rate"]["mean"],
            "agent_scope_lookups_n": configs["cache_only"]["n"]["cache_lookups"],
            "pipeline_scope_lookups_n": configs["cache_only_pipeline"]["n"]["cache_lookups"],
            "agent_scope_entity_mismatch_share": agent_cc.get("entity_mismatch_share_of_hits"),
            "pipeline_scope_entity_mismatch_share": pipe_cc.get(
                "entity_mismatch_share_of_hits"
            ),
        }
    effect = {c: v["cache_early_stop"] for c, v in configs.items() if v.get("cache_early_stop")}
    if effect:
        out["E4_cache_induced_early_stops"] = {
            "description": "early stops that fire only because cache-served drafts are counted "
                           "(not a paper hypothesis; fourth composition effect)",
            "by_config": effect,
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
                "paired": compare(records, heur, lrn),
            }
    if learned:
        out["learned_vs_heuristic_routing"] = learned
    h3 = _h3_test(records, configs)
    if h3 is not None:
        out["H3_routing_success_drop"] = h3
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


def _mechanism_markdown(cfgs: dict[str, Any], order: list[str]) -> list[str]:
    rows = [c for c in order
            if (cfgs[c].get("cache_correctness") or {}).get("mechanisms")]
    if not rows:
        return []
    lines = [
        "", "## Cache hits by mechanism", "",
        "Where each served entry was written. Same loop: earlier in the reasoning loop it is "
        "served back into. Same task: elsewhere in the same task run. Cross task: during an "
        "earlier task, the reuse a semantic cache is meant for. Cross role: written by another "
        "role (for example a plan served to the aggregator). Query-entity mismatch compares "
        "entities in the question or subtask only. Shares are of hits (n = hits); refusals are "
        "of lookups; bypassed requests skipped the cache under `first_step_only`.", "",
        "| Config | Same loop | Same task | Cross task | Cross role | Query-entity mismatch "
        "| Guard refusals | Bypassed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in rows:
        m = cfgs[c]["cache_correctness"]["mechanisms"]
        o = m["share_of_hits"]
        lines.append(
            f"| {c} | {_r(o['same_loop'])} | {_r(o['same_task'])} | {_r(o['cross_task'])} "
            f"| {_r(m['cross_role_share_of_hits'])} "
            f"| {_r(m['query_entity_mismatch_share_of_hits'])} "
            f"| {_r(m['entity_guard_refusals'])} | {m['bypassed_requests']} |"
        )
    lines += ["", "Success given a hit, by origin (n = hits of that origin).", "",
              "| Config | Same loop | Same task | Cross task |", "| --- | --- | --- | --- |"]
    for c in rows:
        sg = cfgs[c]["cache_correctness"]["mechanisms"]["success_given_hit"]
        lines.append(f"| {c} | {_r(sg['same_loop'])} | {_r(sg['same_task'])} "
                     f"| {_r(sg['cross_task'])} |")
    return lines


def _subtask_markdown(cfgs: dict[str, Any], order: list[str]) -> list[str]:
    rows = [c for c in order
            if (cfgs[c].get("cache_correctness") or {}).get("subtask_cache")]
    if not rows:
        return []
    lines = [
        "", "## Subtask cache", "",
        "Resolved subtask answers reused across subtasks. A hit skips the subtask's retrieval and "
        "reasoning. n is subtask lookups for the hit rate, hits for the origin shares, and task "
        "runs for the success columns.", "",
        "| Config | Hit rate | Same task | Cross task | Task success, any subtask hit "
        "| Task success, none |", "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in rows:
        sc = cfgs[c]["cache_correctness"]["subtask_cache"]
        o = sc["share_of_hits"]
        lines.append(f"| {c} | {_r(sc['hit_rate'])} | {_r(o['same_task'])} "
                     f"| {_r(o['cross_task'])} | {_r(sc['task_success_with_subtask_hit'])} "
                     f"| {_r(sc['task_success_without'])} |")
    return lines


def _routing_scores_markdown(cfgs: dict[str, Any], order: list[str]) -> list[str]:
    rows = [c for c in order if cfgs[c].get("routing_scores")]
    if not rows:
        return []
    cut = ", ".join(str(c) for c in RUNG_CUTOFFS)
    lines = [
        "", "## Routing scores", "",
        f"Complexity score per request by role, against the ladder cutoffs ({cut}). A request "
        "goes to the first rung whose cutoff its score is below; the top rung needs a score of "
        "at least the last cutoff. n is requests.", "",
        "| Config | Role | n | Median | P95 | Max | >= 0.25 | >= 0.5 | >= 0.75 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in rows:
        for role, d in cfgs[c]["routing_scores"].items():
            sh = d["share_at_or_above"]
            lines.append(
                f"| {c} | {role} | {d['n']} | {d['median']:.3f} | {d['p95']:.3f} "
                f"| {d['max']:.3f} | " + " | ".join(f"{100 * sh[str(k)]:.1f}%"
                                                     for k in RUNG_CUTOFFS) + " |"
            )
    return lines


def _e4_markdown(e4: dict[str, Any] | None) -> list[str]:
    if not e4:
        return []
    lines = [
        "", "## Composition effect 4: cache-induced early stops", "",
        "A cache-induced early stop fires on the full draft list but would not fire with "
        "cache-served drafts removed. Pooled over seeds. Configs with "
        "`early_stopping.exclude_cached_drafts: true` have 0 by construction.", "",
        "| Config | Induced stops (per task) | Share of early stops | Cached share of drafts "
        "| Replays among cached drafts | Task success, induced stop | Task success, none |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c, e in e4["by_config"].items():
        lines.append(
            f"| {c} | {e['cache_induced_early_stops']} ({e['per_task']:.2f}, n={e['n_tasks']}) "
            f"| {_r(e['share_of_early_stops'])} | {_r(e['cached_share_of_drafts'])} "
            f"| {_r(e['replay_share_of_cached_drafts'])} "
            f"| {_r(e['task_success_with_induced_stop'])} "
            f"| {_r(e['task_success_without_induced_stop'])} |"
        )
    return lines


def _h1_markdown(h1: dict[str, Any] | None, title: str = "H1: composition of caching and routing"
                 ) -> list[str]:
    if not h1:
        return []
    p, s = h1["primary_independence_test"], h1["secondary_sum_of_savings"]
    cache = h1.get("cache_config", "cache_only")
    lines = [
        "", f"## {title}", "",
        f"Compared config: `{h1['compared_config']}`, n = {h1['n_tasks']} tasks "
        f"(paired, all seeds per task). Savings are relative to `baseline` cost.", "",
        "| Quantity | Value |", "| --- | --- |",
        f"| Saving, {cache} | {100 * h1['saving_cache_only']:.1f}% |",
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
        "The sum of savings is secondary: two positive savings compose below their sum even "
        "when independent, and with a negative saving the sum says nothing about interference.",
    ]
    if h1.get("note"):
        lines += ["", f"Note: {h1['note']}."]
    return lines


def _fmt_n(ms: dict[str, float], n: int, pct: bool = False) -> str:
    return f"{_fmt(ms, pct=pct)} (n={n})"


def _p(x: float) -> str:
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def _comparisons_markdown(comparisons: list[dict[str, Any]]) -> list[str]:
    if not comparisons:
        return []
    lines = [
        "## Paired comparisons", "",
        "Each row compares a config with a reference on the tasks and seeds both ran. Success: "
        "exact McNemar test on (task, seed) pairs, and the success-rate difference with a 95% "
        "paired bootstrap CI over tasks. Cost and latency: per-task means over seeds, difference "
        "(config - reference) with a 95% paired bootstrap CI over tasks "
        f"({N_BOOT:,} resamples, seed {BOOT_SEED}).", "",
        "| Config | Reference | Success: ref -> config | Success diff (95% CI) "
        "| McNemar p (discordant ref/config, n pairs) | Cost diff / task (95% CI) "
        "| Cost relative (95% CI) | Latency diff, s (95% CI) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in comparisons:
        s, cost, lat, mc = c["success"], c["cost_usd"], c["latency_s"], c["success"]["mcnemar"]
        rel = "–" if cost["relative"] is None else (
            f"{100 * cost['relative']:+.1f}% {_pct_ci(cost['relative_ci95'])}"
        )
        lines.append(
            f"| {c['config']} | {c['reference']} "
            f"| {100 * s['reference_mean']:.1f}% -> {100 * s['config_mean']:.1f}% "
            f"(n={s['n_tasks']} tasks) "
            f"| {100 * s['delta']:+.1f} pp {_pct_ci(s['delta_ci95'])} "
            f"| {_p(mc['p_value'])} ({mc['reference_only_success']}/{mc['config_only_success']}, "
            f"n={mc['n_pairs']}) "
            f"| ${cost['delta']:+.4f} [{cost['delta_ci95'][0]:+.4f}, {cost['delta_ci95'][1]:+.4f}] "
            f"| {rel} "
            f"| {lat['delta']:+.2f} [{lat['delta_ci95'][0]:+.2f}, {lat['delta_ci95'][1]:+.2f}] |"
        )
    return lines


def _scoring_markdown(cfgs: dict[str, Any], order: list[str],
                      comparisons: list[dict[str, Any]]) -> list[str]:
    rows = [c for c in order if cfgs[c].get("scoring")]
    if not rows:
        return []
    lines = [
        "", "## Scoring: primary vs strict", "",
        "Primary success accepts a numeric answer followed by unit words the question names "
        "(\"95 USD\" for a question \"in USD\"). Strict success is GAIA-style quasi-exact match, "
        "where any trailing unit fails. Pooled over seeds; n is task runs.", "",
        "| Config | Primary success | Strict success | Runs scored differently |",
        "| --- | --- | --- | --- |",
    ]
    for c in rows:
        s = cfgs[c]["scoring"]
        lines.append(f"| {c} | {_r(s['primary'])} | {_r(s['strict'])} "
                     f"| {s['scored_differently']} |")
    strict = [c for c in comparisons if c.get("success_strict")]
    if strict:
        lines += [
            "", "Paired success comparisons under each rule. A conclusion that holds under one "
            "rule but not the other depends on answer formatting.", "",
            "| Config | Reference | Primary diff (95% CI) | McNemar p | Strict diff (95% CI) "
            "| McNemar p |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for c in strict:
            p_, s_ = c["success"], c["success_strict"]
            lines.append(
                f"| {c['config']} | {c['reference']} "
                f"| {100 * p_['delta']:+.1f} pp {_pct_ci(p_['delta_ci95'])} "
                f"| {_p(p_['mcnemar']['p_value'])} "
                f"| {100 * s_['delta']:+.1f} pp {_pct_ci(s_['delta_ci95'])} "
                f"| {_p(s_['mcnemar']['p_value'])} |"
            )
    return lines


def _h3_markdown(h3: dict[str, Any] | None) -> list[str]:
    if not h3:
        return []
    lines = [
        "", "## H3: routing success drop by complexity", "",
        "Drop = success without routing minus success with routing, per task averaged over "
        "seeds. 95% CIs from a bootstrap that resamples tasks within each stratum.", "",
        "| Setting | Stratum | Success without routing | Success with routing "
        "| Drop (95% CI) |",
        "| --- | --- | --- | --- | --- |",
    ]
    settings = [(k, v) for k, v in h3.items() if isinstance(v, dict) and "by_stratum" in v]
    for name, res in settings:
        for cx, d in res["by_stratum"].items():
            lines.append(
                f"| {name} ({res['routed_config']} vs {res['baseline_config']}) | {cx} "
                f"| {100 * d['baseline_success']:.1f}% (n={d['n_tasks']}) "
                f"| {100 * d['routed_success']:.1f}% (n={d['n_tasks']}) "
                f"| {100 * d['drop']:+.1f} pp {_pct_ci(d['drop_ci95'])} |"
            )
    lines += ["", "| Setting | High-stratum drop minus low-stratum drop (95% CI) |",
              "| --- | --- |"]
    for name, res in settings:
        if "high_minus_low" in res:
            d = res["high_minus_low"]
            lines.append(f"| {name} | {100 * d['estimate']:+.1f} pp {_pct_ci(d['ci95'])} |")
    if "multi_minus_single" in h3:
        d = h3["multi_minus_single"]
        lines += ["", "Multi-agent minus single-agent difference-of-differences: "
                  f"{100 * d['estimate']:+.1f} pp, 95% CI {_pct_ci(d['ci95'])}."]
    return lines


def to_markdown(summary: dict[str, Any]) -> str:
    cfgs = summary["configs"]
    order = [c for c in ("baseline", "cache_only", "route_only", "composed") if c in cfgs]
    order += sorted(c for c in cfgs if c not in order)
    hyp = summary["hypotheses"]
    lines = ["# Results", ""]
    lines += _comparisons_markdown(summary.get("comparisons", []))
    lines += _scoring_markdown(cfgs, order, summary.get("comparisons", []))
    lines += _h1_markdown(hyp.get("H1_sub_additive"))
    lines += _h1_markdown(hyp.get("H1_guarded_cache"),
                          "H1 with the guarded cache: composition once false hits are removed")
    lines += _h1_markdown(hyp.get("H1_structured_cache"),
                          "H1 with the structured cache")
    lines += _h3_markdown(hyp.get("H3_routing_success_drop"))
    lines += _e4_markdown(hyp.get("E4_cache_induced_early_stops"))
    lines += _cache_markdown(cfgs, order)
    lines += _mechanism_markdown(cfgs, order)
    lines += _subtask_markdown(cfgs, order)
    lines += _routing_scores_markdown(cfgs, order)
    lines += [
        "", "## Per-config descriptives", "",
        "Mean ± std across seeds. Cost is USD per task at each model's own price. n is task runs "
        "(tasks x seeds) for success and cache lookups for the cache hit rate.",
        "",
        "| Config | Cost / task | P50 latency (s) | P95 latency (s) | Success | Cache hit "
        "| LLM calls / task |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in order:
        r, n = cfgs[c], cfgs[c]["n"]
        hit = _fmt_n(r["cache_hit_rate"], n["cache_lookups"], pct=True) \
            if n["cache_lookups"] else "–"
        lines.append(
            f"| {c} | {_fmt(r['cost_usd'], money=True)} | {_fmt(r['latency_p50_s'])} "
            f"| {_fmt(r['latency_p95_s'])} | {_fmt_n(r['success_rate'], n['task_runs'], True)} "
            f"| {hit} | {_fmt(r['llm_calls_per_task'])} |"
        )
    lines += ["", "## Success by complexity", "",
              "Mean ± std across seeds; n is task runs in the stratum (tasks x seeds).", "",
              "| Config | low | medium | high |", "| --- | --- | --- | --- |"]
    for c in order:
        sbc, n = cfgs[c]["success_by_complexity"], cfgs[c]["n"]["by_complexity"]
        cells = [_fmt_n(sbc[k], n.get(k, 0), pct=True) if k in sbc else "–"
                 for k in ("low", "medium", "high")]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "## Routing share", ""]
    for c in order:
        routing = cfgs[c]["routing_share"].items()
        share = ", ".join(f"{m}: {100 * v['mean']:.0f}%" for m, v in routing)
        lines.append(f"- {c} (n={cfgs[c]['n']['routed_requests']} requests): {share}")
    learners = [c for c in order if cfgs[c].get("routing_policy") == "linucb"]
    if learners:
        lines += ["", "## Learned router: learning curve", "",
                  "Task stream split into quarters (per seed, then averaged). n is task runs "
                  "in the quarter across seeds.", "",
                  "| Config | Quarter | Success | Cost / task |", "| --- | --- | --- | --- |"]
        for c in learners:
            ns = cfgs[c]["n"]["learning_curve_task_runs"]
            for i, q in enumerate(cfgs[c]["learning_curve"], 1):
                lines.append(f"| {c} | {i} | {_fmt_n(q['success_rate'], ns[i - 1], pct=True)} "
                             f"| {_fmt(q['cost_usd'], money=True)} |")
    lines += ["", "## Hypothesis checks (raw)", "", "```json"]
    lines += [json.dumps(hyp, indent=2), "```", ""]
    return "\n".join(lines)
