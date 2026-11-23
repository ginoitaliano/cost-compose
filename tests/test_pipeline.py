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
    for name in ("composed", "single_agent_baseline", "single_agent_route"):
        a = run_task(_runtime(name), task, engine="native")
        b = run_task(_runtime(name), task, engine="langgraph")
        assert b["error"] is None, (name, b["error"])
        for k in ("prediction", "llm_calls", "cache_hits", "routing", "subtasks",
                  "early_stops", "cache_induced_early_stops", "entity_mismatch_hits"):
            assert a[k] == b[k], (name, k)


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


def test_cache_correctness_reported_per_config():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        args = ["run", "--dry-run", "--config-dir", str(CONFIGS), "--tasks", str(SAMPLE_TASKS),
                "--corpus", str(SAMPLE_CORPUS), "--seeds", "0", "--out", str(out),
                "--configs", "baseline", "cache_only",
                str(CONFIGS / "ablations" / "cache_only_pipeline.yaml")]
        assert main(args) == 0
        rows = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines()]
        assert all("entity_mismatch_hits" in r for r in rows)
        cfgs = json.loads((out / "summary.json").read_text())["configs"]
        assert cfgs["baseline"]["cache_correctness"] is None
        cc = cfgs["cache_only"]["cache_correctness"]
        assert cc["scope"] == "agent"
        assert cfgs["cache_only_pipeline"]["cache_correctness"]["scope"] == "pipeline"
        n_lookups = sum(r["requests"] for r in rows if r["config"] == "cache_only")
        assert cc["hit_rate"]["n"] == n_lookups
        assert cc["success_given_hit"]["n"] + cc["success_given_miss"]["n"] == n_lookups
        assert "## Cache correctness" in (out / "summary.md").read_text()


def test_skip_tool_calls_in_pipeline():
    task = Task("t1", "Charter teams get 192 GPU-hours a week. What is 192 * 3.2?", "614.4", "high")
    assert cfg("composed").caching.skip_tool_calls is False
    rt = _runtime("composed_skip_tool_calls")
    first = run_task(rt, task, engine="native")
    second = run_task(rt, task, engine="native")
    assert first["tool_invocations"] >= 1 and first["tool_call_responses_not_cached"] >= 1
    # The repeat can skip the tool entirely when a stored text draft is close enough; what the
    # guard guarantees is that no response carrying tool-call arguments is stored or served.
    assert second["cache_hits"] > 0  # plans, retrieval and text are still cached
    assert sum(rt.cache.stats.tool_call_entries_refused.values()) == 0
    stored = [p for ps in rt.cache.backend._payloads.values() for p in ps]
    assert stored and not any(
        tc["arguments"] for p in stored for tc in p["response"]["tool_calls"]
    )

    loose = _runtime("composed")
    run_task(loose, task, engine="native")
    stored = [p for ps in loose.cache.backend._payloads.values() for p in ps]
    assert any(tc["arguments"] for p in stored for tc in p["response"]["tool_calls"])


def test_single_agent_mode_runs_one_agent_through_the_choke_point():
    task = Task("t1", "Charter teams get 192 GPU-hours a week. What is 192 * 3.2?", "614.4", "high")
    rec = run_task(_runtime("single_agent_baseline"), task, engine="native")
    assert rec["error"] is None
    assert set(rec["requests_by_agent"]) == {"single_agent"}  # no orchestrator, no retriever
    assert rec["subtasks"] == 1 and rec["tool_invocations"] >= 1
    assert rec["prediction"] and rec["cost_usd"] > 0
    assert set(rec["routing"]) == {"gpt-4o"}

    routed = run_task(_runtime("single_agent_route"), task, engine="native")
    assert routed["error"] is None and "gpt-4o" not in routed["routing"]
    assert routed["cost_usd"] < rec["cost_usd"]


def test_single_agent_h3_comparison_in_report():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        abl = CONFIGS / "ablations"
        args = ["run", "--dry-run", "--config-dir", str(CONFIGS), "--tasks", str(SAMPLE_TASKS),
                "--corpus", str(SAMPLE_CORPUS), "--seeds", "0", "--out", str(out),
                "--configs", "baseline", "route_only", str(abl / "single_agent_baseline.yaml"),
                str(abl / "single_agent_route.yaml")]
        assert main(args) == 0
        rows = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines()]
        assert {r["mode"] for r in rows if r["config"].startswith("single")} == {"single_agent"}
        h3 = json.loads((out / "summary.json").read_text())["hypotheses"]["H3_routing_success_drop"]
        assert set(h3) == {"multi_agent", "single_agent", "multi_minus_single"}
        assert h3["single_agent"]["by_stratum"]["high"]["n_tasks"] == 3
