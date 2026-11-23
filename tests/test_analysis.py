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
