"""Pipeline cache guards and hit-mechanism accounting.

The embedder here maps every text to the same vector, so any stored entry is a candidate for
any lookup. That isolates the guards from embedding similarity: what is served is decided only
by namespaces, loop positions and entity checks.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from helpers import CONFIGS, SAMPLE_CORPUS, SAMPLE_TASKS, cfg

from costcompose.agents import Runtime, aggregate_node, plan_node, reason_node
from costcompose.analysis import summarize
from costcompose.cache import InMemoryBackend, SemanticCache
from costcompose.cli import main
from costcompose.early_stop import EarlyStopper
from costcompose.llm import LLMResponse, ToolCall, Usage
from costcompose.retrieval import VectorStore
from costcompose.routing import make_router

GUARDS = ("role_scoped", "first_step_only", "entity_guard", "skip_tool_calls")


class SameVector:
    def embed(self, texts):
        return np.ones((len(texts), 8)) / np.sqrt(8), 1


class Scripted:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def complete(self, model, messages, tools=None, temperature=0.0, seed=None, **_):
        self.calls += 1
        return next(self.responses)


def _rt(name, responses, **guards):
    c = cfg(name)
    for k, v in guards.items():
        setattr(c.caching, k, v)
    emb = SameVector()
    cache = SemanticCache(InMemoryBackend(), emb, c.caching.threshold, c.caching.scope,
                          c.caching.model_aware, skip_tool_calls=c.caching.skip_tool_calls,
                          role_scoped=c.caching.role_scoped, entity_guard=c.caching.entity_guard)
    return Runtime(cfg=c, llm=Scripted(responses), router=make_router(c.routing),
                   stopper=EarlyStopper(c.early_stopping), store=VectorStore(emb), cache=cache,
                   seed=0)


def _text(s):
    return LLMResponse(s, "gpt-4o", Usage(10, 10))


def test_defaults_keep_the_naive_cache():
    for name in ("cache_only", "composed", "cache_route", "cache_only_pipeline"):
        c = cfg(name)
        assert not any(getattr(c.caching, g) for g in GUARDS), name


def test_guarded_configs_turn_every_guard_on():
    for name in ("cache_only_guarded", "composed_guarded", "cache_route_guarded"):
        assert all(getattr(cfg(name).caching, g) for g in GUARDS), name


def test_naive_cache_serves_the_plan_to_the_aggregator():
    plan = '["find x", "find y"]'
    rt = _rt("cache_only", [_text(plan), _text("FINAL ANSWER: 7")])
    state = plan_node(rt, {"question": "What is x?"})
    state.update(question="What is x?", outputs=["7"])
    out = aggregate_node(rt, state)
    assert out["final"] == plan  # the stored plan came back as the final answer
    agg = rt.trace.calls[-1]
    assert agg.cache_hit and agg.role == "aggregator"
    assert agg.cross_role_hit and agg.hit_origin == "same_task"


def test_role_scoped_cache_keeps_plan_and_answer_apart():
    rt = _rt("cache_only", [_text('["find x"]'), _text("FINAL ANSWER: 7")], role_scoped=True)
    state = plan_node(rt, {"question": "What is x?"})
    state.update(question="What is x?", outputs=["7"])
    assert aggregate_node(rt, state)["final"] == "FINAL ANSWER: 7"
    assert not rt.trace.calls[-1].cache_hit


STATE = {"question": "What is 2*3?", "subtasks": ["compute 2*3"], "index": 0, "outputs": [],
         "context": "passages"}


def _loop_script():
    call = LLMResponse("", "gpt-4o", Usage(1, 1), [ToolCall("c1", "calculator",
                                                             {"expression": "2*3"})])
    final = _text("ANSWER: 6\nCONFIDENCE: 0.9\nSTATUS: FINAL")
    return [call, final, final, final, final]


def test_naive_cache_replays_a_loop_step_into_its_own_loop():
    rt = _rt("cache_only", _loop_script())
    reason_node(rt, dict(STATE))
    hits = [c for c in rt.trace.calls if c.cache_hit]
    assert hits and all(c.hit_origin == "same_loop" for c in hits)
    assert rt.trace.tool_invocations > 1  # the calculator ran again for each replayed call
    s = rt.trace.summary()
    assert s["hits_by_origin"] == {"same_loop": len(hits)}


def test_first_step_only_bypasses_later_loop_steps():
    rt = _rt("cache_only", _loop_script(), first_step_only=True)
    out = reason_node(rt, dict(STATE))
    assert out["outputs"] == ["6"]
    assert rt.trace.tool_invocations == 1
    s = rt.trace.summary()
    assert s["cache_hits"] == 0
    assert s["cache_lookups"] == 1 and s["cache_bypassed"] == 1


def test_entity_guard_refuses_a_hit_for_a_different_id():
    rt = _rt("cache_only", [_text("FINAL ANSWER: shipped"), _text("FINAL ANSWER: pending")],
             entity_guard=True)
    for q in ("Status of order ORD-1001?", "Status of order ORD-2002?"):
        aggregate_node(rt, {"question": q, "subtasks": ["s"], "outputs": ["x"]})
    last = rt.trace.calls[-1]
    assert not last.cache_hit and last.guard_refused
    assert rt.trace.summary()["entity_guard_refusals"] == 1


def test_without_the_guard_the_mismatch_is_served_and_counted():
    rt = _rt("cache_only", [_text("FINAL ANSWER: shipped")])
    for q in ("Status of order ORD-1001?", "Status of order ORD-2002?"):
        aggregate_node(rt, {"question": q, "subtasks": ["s"], "outputs": ["x"]})
    last = rt.trace.calls[-1]
    assert last.cache_hit and last.query_entity_mismatch and not last.cross_role_hit


def test_hits_from_an_earlier_task_are_cross_task():
    rt = _rt("cache_only", [_text("FINAL ANSWER: 1")])
    aggregate_node(rt, {"question": "q", "subtasks": ["s"], "outputs": ["x"]})
    rt.trace = type(rt.trace)()  # what run_task does between tasks
    aggregate_node(rt, {"question": "q", "subtasks": ["s"], "outputs": ["x"]})
    assert rt.trace.calls[-1].hit_origin == "cross_task"


def test_report_has_mechanisms_routing_scores_and_guarded_h1():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        abl = CONFIGS / "ablations"
        configs = ["baseline", "cache_only", "route_only", "composed",
                   str(abl / "cache_route.yaml"), str(abl / "cache_only_guarded.yaml"),
                   str(abl / "cache_route_guarded.yaml")]
        assert main(["run", "--dry-run", "--config-dir", str(CONFIGS), "--tasks",
                     str(SAMPLE_TASKS), "--corpus", str(SAMPLE_CORPUS), "--seeds", "0",
                     "--out", str(out), "--configs", *configs]) == 0
        summary = json.loads((out / "summary.json").read_text())
        md = (out / "summary.md").read_text()
    mech = summary["configs"]["cache_only"]["cache_correctness"]["mechanisms"]
    assert set(mech["share_of_hits"]) == {"same_loop", "same_task", "cross_task"}
    assert summary["configs"]["route_only"]["routing_scores"]
    assert summary["configs"]["baseline"]["routing_scores"] is None
    h1g = summary["hypotheses"]["H1_guarded_cache"]
    assert h1g["cache_config"] == "cache_only_guarded"
    assert "## Cache hits by mechanism" in md and "## Routing scores" in md
    assert "H1 with the guarded cache" in md


def test_old_records_without_mechanism_fields_still_summarize():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        assert main(["run", "--dry-run", "--config-dir", str(CONFIGS), "--tasks",
                     str(SAMPLE_TASKS), "--corpus", str(SAMPLE_CORPUS), "--seeds", "0",
                     "--out", str(out), "--configs", "baseline", "cache_only",
                     "route_only"]) == 0
        rows = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines()]
    new = ("cache_lookups", "cache_bypassed", "hits_by_origin", "hits_by_role", "lookups_by_role",
           "cross_role_hits", "query_entity_mismatch_hits", "entity_guard_refusals",
           "routing_scores")
    old = [{k: v for k, v in r.items() if k not in new} for r in rows]
    s = summarize(old)
    assert s["configs"]["cache_only"]["cache_correctness"]["mechanisms"] is None
    assert s["configs"]["route_only"]["routing_scores"] is None
