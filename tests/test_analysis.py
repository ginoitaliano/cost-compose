import numpy as np

from costcompose import analysis


def _records(factors: dict[str, float], n: int = 40, seeds=(0, 1, 2), noise: float = 0.0):
    """Synthetic records: each config costs factor * baseline cost of the same task."""
    rng = np.random.default_rng(1)
    base = rng.uniform(0.01, 0.10, size=n)
    out = []
    for cfg, f in factors.items():
        for seed in seeds:
            for t in range(n):
                jitter = 1.0 + noise * rng.standard_normal() if noise else 1.0
                out.append({"config": cfg, "seed": seed, "task_id": f"t{t}",
                            "cost_usd": float(base[t] * f * jitter)})
    return out


def test_h1_independent_composition_has_zero_interaction():
    recs = _records({"baseline": 1.0, "cache_only": 0.7, "route_only": 0.6, "cache_route": 0.42})
    h1 = analysis._h1_test(recs, "cache_route")
    p = h1["primary_independence_test"]
    assert h1["n_tasks"] == 40
    assert abs(p["independent_prediction"] - (1 - 0.7 * 0.6)) < 1e-9
    assert abs(p["interaction"]) < 1e-9
    assert p["interaction_ci95"][0] <= 1e-9 and p["interaction_ci95"][1] >= -1e-9
    # Against the sum (0.3 + 0.4 = 0.7) the same independent result looks sub-additive.
    assert h1["secondary_sum_of_savings"]["observed_over_additive"] < 1.0


def test_h1_interference_is_negative_with_ci_below_zero():
    recs = _records({"baseline": 1.0, "cache_only": 0.7, "route_only": 0.6, "cache_route": 0.5},
                    noise=0.05)
    p = analysis._h1_test(recs, "cache_route")["primary_independence_test"]
    assert p["interaction"] < 0
    assert p["interaction_ci95"][1] < 0
    assert p["verdict"].startswith("sub-independent")


def test_h1_bootstrap_is_reproducible():
    recs = _records({"baseline": 1.0, "cache_only": 0.7, "route_only": 0.6, "composed": 0.45},
                    noise=0.1)
    a = analysis._h1_test(recs, "composed")
    b = analysis._h1_test(recs, "composed")
    assert a == b


def test_h1_uses_only_tasks_present_in_every_config():
    recs = _records({"baseline": 1.0, "cache_only": 0.7, "route_only": 0.6, "composed": 0.42})
    recs = [r for r in recs if not (r["config"] == "composed" and r["task_id"] == "t0")]
    assert analysis._h1_test(recs, "composed")["n_tasks"] == 39


# -- paired statistics -----------------------------------------------------------------------


def test_mcnemar_exact_known_values():
    # 6 discordant pairs, all favouring one side: p = 2 * (1/2)^6
    r = analysis._mcnemar_exact([False] * 6 + [True] * 4, [True] * 6 + [True] * 4)
    assert r["config_only_success"] == 6 and r["reference_only_success"] == 0
    assert abs(r["p_value"] - 0.03125) < 1e-12 and r["n_pairs"] == 10
    # balanced discordance: p = 1
    assert analysis._mcnemar_exact([True, False], [False, True])["p_value"] == 1.0
    # no discordant pairs
    assert analysis._mcnemar_exact([True, True], [True, True])["p_value"] == 1.0


def _success_records(rates: dict[str, dict[str, float]], per_stratum: int = 40, seeds=(0, 1, 2)):
    """Deterministic success patterns: in each stratum, the first round(rate * n) tasks succeed."""
    out = []
    for cfg, by_cx in rates.items():
        for cx, rate in by_cx.items():
            k = round(rate * per_stratum)
            for t in range(per_stratum):
                for seed in seeds:
                    out.append({"config": cfg, "seed": seed, "task_id": f"{cx}{t}",
                                "complexity": cx, "success": t < k,
                                "cost_usd": 0.01, "latency_s": 1.0})
    return out


def test_compare_reports_paired_success_cost_and_latency():
    recs = _success_records({"baseline": {"low": 0.9, "high": 0.9},
                             "route_only": {"low": 0.9, "high": 0.4}})
    c = analysis.compare(recs, "baseline", "route_only")
    s = c["success"]
    assert s["n_tasks"] == 80 and s["mcnemar"]["n_pairs"] == 240
    assert abs(s["delta"] - (-0.25)) < 1e-9
    assert s["delta_ci95"][1] < 0 and s["mcnemar"]["p_value"] < 0.001
    assert c["cost_usd"]["delta"] == 0 and c["latency_s"]["delta_ci95"] == [0.0, 0.0]


def test_h3_difference_of_differences():
    recs = _success_records({"baseline": {"low": 0.9, "medium": 0.8, "high": 0.7},
                             "route_only": {"low": 0.85, "medium": 0.6, "high": 0.3}})
    h3, _ = analysis._h3_drop(recs, "baseline", "route_only")
    by = h3["by_stratum"]
    assert {k: v["n_tasks"] for k, v in by.items()} == {"low": 40, "medium": 40, "high": 40}
    assert abs(by["low"]["drop"] - 0.05) < 1e-9 and abs(by["high"]["drop"] - 0.40) < 1e-9
    did = h3["high_minus_low"]
    assert abs(did["estimate"] - 0.35) < 1e-9
    assert did["ci95"][0] > 0  # high stratum loses clearly more under routing


def test_report_prints_sample_sizes():
    recs = _success_records({"baseline": {"low": 0.5}, "route_only": {"low": 0.5}},
                            per_stratum=10, seeds=(0,))
    for r in recs:
        r.update(requests=4, cache_hits=0, requests_by_agent={}, cache_hits_by_agent={},
                 routing={"gpt-4o": 4}, cross_model_hits=0, cost_reference_usd=0.01,
                 llm_calls=4, early_stops=0, reasoning_iterations=1, techniques={})
    md = analysis.to_markdown(analysis.summarize(recs))
    assert "50.0 ± 0.0% (n=10)" in md  # success rate with task runs
    assert "(n=40 requests)" in md  # routing share
    assert "## Paired comparisons" in md


def test_h3_multi_minus_single_is_paired():
    recs = _success_records({
        "baseline": {"low": 0.9, "high": 0.8}, "route_only": {"low": 0.85, "high": 0.4},
        "single_agent_baseline": {"low": 0.9, "high": 0.8},
        "single_agent_route": {"low": 0.85, "high": 0.7},
    })
    configs = {c: {} for c in ("baseline", "route_only", "single_agent_baseline",
                               "single_agent_route")}
    h3 = analysis._h3_test(recs, configs)
    assert abs(h3["multi_agent"]["high_minus_low"]["estimate"] - 0.35) < 1e-9
    assert abs(h3["single_agent"]["high_minus_low"]["estimate"] - 0.05) < 1e-9
    mms = h3["multi_minus_single"]
    assert abs(mms["estimate"] - 0.30) < 1e-9 and mms["ci95"][0] > 0
