import json
import random
import tempfile
from pathlib import Path

from helpers import CONFIGS, SAMPLE_CORPUS, SAMPLE_TASKS, cfg
from test_pipeline import _runtime

from costcompose.cli import main
from costcompose.config import from_dict
from costcompose.pipeline import run_task
from costcompose.routing import (
    ComplexityFeatures,
    ComplexityRouter,
    LinUCBRouter,
    RoutingDecision,
    make_router,
)
from costcompose.tasks import load_tasks


def _two_arm_cfg(alpha=1.0):
    c = cfg("learned_route").routing
    c.ladder = c.ladder[1:3]  # cheap = Haiku, expensive = GPT-4o
    c.bandit.alpha = alpha
    return c


def test_factory_picks_policy():
    assert isinstance(make_router(cfg("learned_route").routing), LinUCBRouter)
    assert isinstance(make_router(cfg("route_only").routing), ComplexityRouter)
    fixed = make_router(cfg("baseline").routing).route(ComplexityFeatures(1, 0, 1, 0))
    assert fixed.tier == "fixed"


def test_linucb_learns_cost_quality_tradeoff():
    """Cheap arm solves shallow subtasks only; expensive arm solves everything but costs more.
    The bandit should learn to send shallow work cheap and deep work expensive."""
    for seed in range(5):
        r = LinUCBRouter(_two_arm_cfg())
        rng = random.Random(seed)
        for _ in range(400):
            depth = rng.choice([0, 3])
            d = r.route(ComplexityFeatures(100, 1, 1000, depth))
            cheap = d.arm == 0
            success = (depth == 0) or not cheap
            r.observe(d, 0.05 if cheap else 1.0, 1.0)
            r.end_task(success)
        assert r.route(ComplexityFeatures(100, 1, 1000, 0)).arm == 0, seed
        assert r.route(ComplexityFeatures(100, 1, 1000, 3)).arm == 1, seed
        state = r.state_dict()
        assert state["policy"] == "linucb" and sum(a["pulls"] for a in state["arms"]) == 400


def test_unobserved_decisions_are_not_learned_from():
    r = LinUCBRouter(_two_arm_cfg())
    r.observe(RoutingDecision("gpt-4o", "fixed", 0.3), 1.0, 1.0)  # arm -1: not a bandit pick
    assert r.end_task(True) == 0


def test_end_task_applies_and_clears_pending():
    r = LinUCBRouter(_two_arm_cfg())
    for _ in range(3):
        r.observe(r.route(ComplexityFeatures(10, 0, 10, 1)), 0.1, 1.0)
    assert r.end_task(True) == 3
    assert r.end_task(True) == 0
    r.reset()
    assert r.pulls.sum() == 0 and not r.distribution


def test_learned_router_updates_inside_pipeline():
    rt = _runtime("learned_route")
    tasks = load_tasks(SAMPLE_TASKS)
    recs = [run_task(rt, t, engine="native") for t in tasks[:3]]
    assert all(r["routing_policy"] == "linucb" for r in recs)
    assert all(r["router_updates"] == r["llm_calls"] for r in recs)
    assert rt.router.pulls.sum() == sum(r["llm_calls"] for r in recs)


def test_cache_hits_do_not_update_router():
    rt = _runtime("composed_learned")
    task = load_tasks(SAMPLE_TASKS)[0]
    run_task(rt, task, engine="native")
    rec = run_task(rt, task, engine="native")
    assert rec["cache_hits"] > 0
    assert rec["router_updates"] == rec["llm_calls"]


def test_cli_writes_policy_state_and_learning_curve():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        args = ["run", "--dry-run", "--config-dir", str(CONFIGS), "--tasks", str(SAMPLE_TASKS),
                "--corpus", str(SAMPLE_CORPUS), "--seeds", "0", "1", "--out", str(out),
                "--configs", "route_only", str(CONFIGS / "ablations" / "learned_route.yaml")]
        assert main(args) == 0
        state = json.loads((out / "policies" / "learned_route_seed0.json").read_text())
        assert state["policy"] == "linucb" and len(state["arms"]) == 4
        summary = json.loads((out / "summary.json").read_text())
        assert len(summary["configs"]["learned_route"]["learning_curve"]) == 4
        learned = summary["hypotheses"]["learned_vs_heuristic_routing"]
        assert "learned_route_vs_route_only" in learned
        assert "Learned router" in (out / "summary.md").read_text()


def test_bad_policy_rejected():
    raw = {"name": "x", "routing": {"policy": "ppo"},
           "prices": {"gpt-4o": {"input_per_mtok": 1, "output_per_mtok": 1},
                      "text-embedding-3-small": {"input_per_mtok": 1, "output_per_mtok": 0}}}
    try:
        from_dict(raw)
    except ValueError as e:
        assert "routing.policy" in str(e)
    else:
        raise AssertionError("expected ValueError")
