import importlib.util
import json
import tempfile
from pathlib import Path

from helpers import CONFIGS, SAMPLE_CORPUS, SAMPLE_TASKS, cfg

from costcompose.agents import Runtime, parse_plan
from costcompose.cli import main
from costcompose.early_stop import EarlyStopper
from costcompose.experiment import make_cache
from costcompose.llm import FakeLLMClient, HashingEmbedder
from costcompose.pipeline import run_task
from costcompose.retrieval import VectorStore
from costcompose.routing import make_router
from costcompose.tasks import Task, load_tasks


def _runtime(name):
    c = cfg(name)
    emb = HashingEmbedder()
    return Runtime(
        cfg=c,
        llm=FakeLLMClient(),
        router=make_router(c.routing),
        stopper=EarlyStopper(c.early_stopping),
        store=VectorStore.from_directory(SAMPLE_CORPUS, emb),
        cache=make_cache(c, emb, "test"),
        seed=0,
    )


def test_parse_plan():
    assert parse_plan('Sure: ["a", "b"]', "q", 4) == ["a", "b"]
    assert parse_plan("not json", "q", 4) == ["q"]
    assert parse_plan('["a","b","c","d","e"]', "q", 2) == ["a", "b"]


def test_native_run_produces_complete_record():
    rt = _runtime("composed")
    task = Task("t1", "Charter teams get 192 GPU-hours a week. What is 192 * 3.2?", "614.4", "high")
    rec = run_task(rt, task, engine="native")
    assert rec["error"] is None
    assert rec["llm_calls"] >= 4  # plan + retrieve + reason + aggregate
    assert rec["cost_usd"] > 0
    assert rec["tool_invocations"] >= 1  # fake reasoner calls the calculator on arithmetic
    assert set(rec["requests_by_agent"]) == {"orchestrator", "retriever", "reasoner"}


def test_cache_hits_on_repeated_task():
    rt = _runtime("cache_only")
    task = load_tasks(SAMPLE_TASKS)[0]
    first = run_task(rt, task, engine="native")
    second = run_task(rt, task, engine="native")
    assert second["cache_hits"] > first["cache_hits"]
    assert second["llm_calls"] < first["llm_calls"]
    assert second["cache_hits_by_agent"]["orchestrator"] == 2  # plan + aggregate replayed
    assert second["cost_usd"] < first["cost_usd"]


def test_routing_uses_cheaper_models():
    task = load_tasks(SAMPLE_TASKS)[0]
    base = run_task(_runtime("baseline"), task, engine="native")
    routed = run_task(_runtime("route_only"), task, engine="native")
    assert set(base["routing"]) == {"gpt-4o"}
    assert routed["cost_usd"] < base["cost_usd"]


def test_langgraph_engine_matches_native():
    if importlib.util.find_spec("langgraph") is None:
        return  # optional dependency; exercised in CI's "full" job
    task = load_tasks(SAMPLE_TASKS)[4]
    a = run_task(_runtime("composed"), task, engine="native")
    b = run_task(_runtime("composed"), task, engine="langgraph")
    assert b["error"] is None, b["error"]
    for k in ("prediction", "llm_calls", "cache_hits", "routing", "subtasks"):
        assert a[k] == b[k], k


def test_cli_dry_run_end_to_end_and_resume():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        args = ["run", "--dry-run", "--config-dir", str(CONFIGS), "--tasks", str(SAMPLE_TASKS),
                "--corpus", str(SAMPLE_CORPUS), "--seeds", "0", "1", "--out", str(out)]
        assert main(args) == 0
        rows = (out / "runs.jsonl").read_text().splitlines()
        assert len(rows) == 4 * 2 * 9
        summary = json.loads((out / "summary.json").read_text())
        assert set(summary["configs"]) == {"baseline", "cache_only", "route_only", "composed"}
        assert "H1_sub_additive" in summary["hypotheses"]
        assert (out / "summary.md").read_text().startswith("# Results")

        assert main(args) == 0  # resume: nothing new to run
        assert len((out / "runs.jsonl").read_text().splitlines()) == len(rows)


def test_cli_validate():
    assert main(["validate", "--config-dir", str(CONFIGS), "--tasks", str(SAMPLE_TASKS)]) == 0
