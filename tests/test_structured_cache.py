"""Structured cache keys and the resolved-subtask cache.

The embedder maps every text to the same vector, so semantic similarity never decides anything
here: what is served depends only on roles and on the exact dependency digests.
"""

import json
import tempfile
from pathlib import Path

from helpers import CONFIGS, SAMPLE_CORPUS, SAMPLE_TASKS, cfg
from test_cache_guards import SameVector, _loop_script, _rt, _text

from costcompose.agents import Runtime, aggregate_node, plan_node, reason_node
from costcompose.cli import main
from costcompose.early_stop import EarlyStopper
from costcompose.experiment import make_cache
from costcompose.llm import FakeLLMClient, HashingEmbedder
from costcompose.pipeline import run_task
from costcompose.retrieval import VectorStore
from costcompose.routing import make_router
from costcompose.tasks import Task
from costcompose.tracking import TaskTrace


def _srt(responses, subtask_cache=False):
    rt = _rt("cache_only", responses, key="structured", subtask_cache=subtask_cache)
    rt.cache.structured = True
    return rt


def _state(sub, outputs=(), docs=("d1",), subtasks=None):
    subs = list(subtasks or [*(f"earlier {i}" for i in range(len(outputs))), sub])
    return {"question": "Q?", "subtasks": subs, "index": len(outputs), "outputs": list(outputs),
            "context": "passages", "context_docs": list(docs)}


FINAL = "ANSWER: 6\nCONFIDENCE: 0.9\nSTATUS: FINAL"


def test_defaults_are_text_keys_without_subtask_cache():
    for name in ("cache_only", "composed", "cache_route", "cache_only_guarded"):
        c = cfg(name).caching
        assert c.key == "text" and not c.subtask_cache, name
    for name in ("cache_only_structured", "composed_structured", "cache_route_structured"):
        c = cfg(name).caching
        assert c.key == "structured" and c.subtask_cache, name


def test_plan_is_never_served_to_the_aggregator():
    rt = _srt([_text('["find x"]'), _text("FINAL ANSWER: 7")])
    state = plan_node(rt, {"question": "What is x?"})
    state.update(question="What is x?", outputs=["7"])
    assert aggregate_node(rt, state)["final"] == "FINAL ANSWER: 7"


def test_loop_steps_are_not_served_their_own_history():
    rt = _srt(_loop_script())
    out = reason_node(rt, _state("compute 2*3"))
    assert out["outputs"] == ["6"] and rt.trace.tool_invocations == 1
    assert not any(c.cache_hit for c in rt.trace.calls)


def test_different_upstream_answers_or_documents_miss():
    rt = _srt([_text(FINAL)] * 3)
    reason_node(rt, _state("who owns it", outputs=["Acme"]))
    reason_node(rt, _state("who owns it", outputs=["Globex"]))  # other upstream answer
    reason_node(rt, _state("who owns it", outputs=["Acme"], docs=("d2",)))  # other documents
    assert not any(c.cache_hit for c in rt.trace.calls)


def test_same_ask_on_same_inputs_in_a_later_task_hits():
    rt = _srt([_text(FINAL)])
    reason_node(rt, _state("who owns The Collegian", outputs=["x"]))
    rt.trace = TaskTrace()
    out = reason_node(rt, _state("who owns The Collegian", outputs=["x"]))
    assert out["outputs"] == ["x", "6"]
    hit = rt.trace.calls[-1]
    assert hit.cache_hit and hit.hit_origin == "cross_task"


def test_entities_of_the_ask_must_match():
    rt = _srt([_text("FINAL ANSWER: shipped"), _text("FINAL ANSWER: pending")])
    for q in ("Status of order ORD-1001?", "Status of order ORD-2002?"):
        aggregate_node(rt, {"question": q, "subtasks": ["s"], "outputs": ["x"]})
    assert not rt.trace.calls[-1].cache_hit


def _pipeline_rt(name):
    c = cfg(name)
    emb = HashingEmbedder()
    return Runtime(cfg=c, llm=FakeLLMClient(), router=make_router(c.routing),
                   stopper=EarlyStopper(c.early_stopping),
                   store=VectorStore.from_directory(SAMPLE_CORPUS, emb),
                   cache=make_cache(c, emb, "t"), seed=0)


def test_subtask_cache_skips_a_resolved_subtask_in_a_later_task():
    rt = _pipeline_rt("cache_only_structured")
    q = "Charter teams get 192 GPU-hours a week. What is 192 * 3.2?"
    first = run_task(rt, Task("a", q, "614.4", "high"), engine="native")
    second = run_task(rt, Task("b", q, "614.4", "high"), engine="native")
    assert first["subtask_cache_hits"] == 0 and first["subtask_cache_lookups"] >= 1
    assert second["subtask_cache_hits"] == second["subtask_cache_lookups"] >= 1
    assert second["subtask_hits_by_origin"] == {"cross_task": second["subtask_cache_hits"]}
    assert second["requests_by_agent"].get("reasoner", 0) == 0  # every subtask skipped
    assert second["prediction"] == first["prediction"]


def test_subtask_cache_respects_upstream_answers():
    rt = _srt([_text(FINAL)] * 2, subtask_cache=True)
    s = _state("who founded it", outputs=["Acme"])
    rt.pending_subtask = ("who founded it", SameVector().embed(["x"])[0][0])
    reason_node(rt, s)  # stores the answer for upstream ["Acme"]
    rt.trace = TaskTrace()
    answer, _ = rt.subtask_lookup(_state("who founded it", outputs=["Globex"]), "who founded it")
    assert answer is None
    answer, _ = rt.subtask_lookup(_state("who founded it", outputs=["Acme"]), "who founded it")
    assert answer == "6"


def test_report_has_subtask_section_and_structured_h1():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        abl = CONFIGS / "ablations"
        configs = ["baseline", "cache_only", "route_only",
                   str(abl / "cache_route.yaml"), str(abl / "cache_only_guarded.yaml"),
                   str(abl / "cache_only_structured.yaml"),
                   str(abl / "cache_route_structured.yaml")]
        assert main(["run", "--dry-run", "--config-dir", str(CONFIGS), "--tasks",
                     str(SAMPLE_TASKS), "--corpus", str(SAMPLE_CORPUS), "--seeds", "0",
                     "--out", str(out), "--configs", *configs]) == 0
        summary = json.loads((out / "summary.json").read_text())
        md = (out / "summary.md").read_text()
    assert summary["hypotheses"]["H1_structured_cache"]["cache_config"] == "cache_only_structured"
    assert "## Subtask cache" in md
    refs = {(c["reference"], c["config"]) for c in summary["comparisons"]}
    assert ("cache_only_guarded", "cache_only_structured") in refs
