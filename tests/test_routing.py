from helpers import cfg

from costcompose.routing import ComplexityFeatures, ComplexityRouter


def test_disabled_router_uses_default_model():
    r = ComplexityRouter(cfg("baseline").routing)
    d = r.route(ComplexityFeatures(5000, 9, 50000, 9))
    assert d.model == "gpt-4o" and d.tier == "fixed"


def test_simple_subtask_goes_low_tier():
    r = ComplexityRouter(cfg("route_only").routing)
    d = r.route(ComplexityFeatures(20, 0, 100, 1))
    assert d.tier == "low"
    assert "Llama" in d.model or "haiku" in d.model


def test_complex_subtask_goes_high_tier():
    r = ComplexityRouter(cfg("route_only").routing)
    d = r.route(ComplexityFeatures(600, 4, 9000, 3))
    assert d.tier == "high" and d.score == 1.0
    assert d.model == "anthropic/claude-sonnet-4-5"


def test_score_is_monotonic_in_each_feature():
    r = ComplexityRouter(cfg("route_only").routing)
    base = ComplexityFeatures(50, 1, 1000, 1)
    s0 = r.score(base)
    for bumped in (
        ComplexityFeatures(200, 1, 1000, 1),
        ComplexityFeatures(50, 3, 1000, 1),
        ComplexityFeatures(50, 1, 5000, 1),
        ComplexityFeatures(50, 1, 1000, 3),
    ):
        assert r.score(bumped) > s0


def test_threshold_boundary_is_high_tier():
    c = cfg("route_only").routing
    r = ComplexityRouter(c)
    # score exactly 0.5: dependency depth and tool calls saturated, the rest zero
    d = r.route(ComplexityFeatures(0, 4, 0, 3))
    assert abs(d.score - 0.5) < 1e-9
    assert d.tier == "high" and d.model == "gpt-4o"


def test_distribution_counts():
    r = ComplexityRouter(cfg("route_only").routing)
    for _ in range(3):
        r.route(ComplexityFeatures(1, 0, 1, 0))
    assert sum(r.distribution.values()) == 3
