"""Cache x early-stop interaction: stops triggered by drafts replayed from the cache."""

from helpers import SAMPLE_TASKS, cfg
from test_pipeline import _runtime

from costcompose.agents import reason_node
from costcompose.llm import LLMResponse, Usage
from costcompose.pipeline import run_task
from costcompose.tasks import load_tasks
from costcompose.tracking import CallRecord


def _scripted(rt, script):
    """Replace the runtime's choke point with a script of (content, served_from_cache)."""
    it = iter(script)

    def call_llm(agent, messages, features, local_input, pipeline_context, use_tools=False):
        content, hit = next(it)
        rt.trace.calls.append(CallRecord(agent, "gpt-4o", "fixed", 0.0, 0, 0, 0.0, 0.0, hit,
                                         False, 0, 0.0))
        return LLMResponse(content, "gpt-4o", Usage(1, 1))

    rt.call_llm = call_llm


STATE = {"question": "q", "subtasks": ["s"], "index": 0, "outputs": [], "context": ""}
FRESH = "ANSWER: 750\nCONFIDENCE: 0.50\nSTATUS: DRAFT"
REPLAY = "ANSWER: 750\nCONFIDENCE: 0.95\nSTATUS: DRAFT"  # same answer, served from cache


def test_replayed_draft_triggering_stop_is_counted():
    rt = _runtime("composed")
    _scripted(rt, [(FRESH, False), (REPLAY, True)])
    reason_node(rt, STATE)
    t = rt.trace
    assert t.early_stops == 1 and t.reasoning_iterations == 2
    assert t.cached_drafts == 1 and t.cached_draft_replays == 1
    assert t.cache_induced_early_stops == 1


def test_stop_reached_by_fresh_drafts_is_not_cache_induced():
    rt = _runtime("composed")
    _scripted(rt, [(FRESH, False), (REPLAY, False)])
    reason_node(rt, STATE)
    assert rt.trace.early_stops == 1 and rt.trace.cache_induced_early_stops == 0


def test_exclude_cached_drafts_keeps_iterating():
    rt = _runtime("composed_exclude_cached_drafts")
    fresh2 = "ANSWER: 750\nCONFIDENCE: 0.95\nSTATUS: DRAFT"
    _scripted(rt, [(FRESH, False), (REPLAY, True), (fresh2, False)])
    reason_node(rt, STATE)
    t = rt.trace
    assert t.reasoning_iterations == 3  # the cached draft did not end the loop
    assert t.early_stops == 1 and t.cache_induced_early_stops == 0


def test_flag_defaults_off_and_ablation_turns_it_on():
    assert cfg("composed").early_stopping.exclude_cached_drafts is False
    assert cfg("composed_exclude_cached_drafts").early_stopping.exclude_cached_drafts is True


def test_repeated_task_in_pipeline():
    task = load_tasks(SAMPLE_TASKS)[0]
    rt = _runtime("composed")
    run_task(rt, task, engine="native")
    rec = run_task(rt, task, engine="native")
    assert rec["cached_drafts"] > 0 and rec["cache_induced_early_stops"] >= 1

    rt = _runtime("composed_exclude_cached_drafts")
    run_task(rt, task, engine="native")
    rec = run_task(rt, task, engine="native")
    assert rec["cached_drafts"] > 0 and rec["cache_induced_early_stops"] == 0
