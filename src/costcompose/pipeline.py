"""Pipeline engines and single-task execution.

Graph: plan -> (retrieve -> reason) x N subtasks -> aggregate

``engine="langgraph"`` compiles that graph with LangGraph (as in the paper; LangSmith tracing is
picked up from the standard LANGSMITH_* environment variables). ``engine="native"`` walks the same
nodes in plain Python with no extra dependencies. ``engine="auto"`` prefers LangGraph when it is
installed.
"""

from __future__ import annotations

import importlib.util
from functools import partial
from typing import Any, TypedDict

from .agents import Runtime, aggregate_node, plan_node, reason_node, retrieve_node
from .evaluation import extract_final_answer, score
from .tasks import Task
from .tracking import TaskTrace


class PipelineState(TypedDict, total=False):
    task_id: str
    question: str
    subtasks: list[str]
    index: int
    outputs: list[str]
    context: str
    final: str


def resolve_engine(engine: str) -> str:
    if engine == "auto":
        return "langgraph" if importlib.util.find_spec("langgraph") else "native"
    if engine not in {"native", "langgraph"}:
        raise ValueError(f"unknown engine {engine!r}")
    return engine


def run_native(rt: Runtime, state: dict[str, Any]) -> dict[str, Any]:
    state = {**state, **plan_node(rt, state)}
    while state["index"] < len(state["subtasks"]):
        state = {**state, **retrieve_node(rt, state)}
        state = {**state, **reason_node(rt, state)}
    return {**state, **aggregate_node(rt, state)}


def build_langgraph(rt: Runtime):
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(PipelineState)
    g.add_node("plan", partial(plan_node, rt))
    g.add_node("retrieve", partial(retrieve_node, rt))
    g.add_node("reason", partial(reason_node, rt))
    g.add_node("aggregate", partial(aggregate_node, rt))
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "reason")
    g.add_conditional_edges(
        "reason",
        lambda s: "retrieve" if s["index"] < len(s["subtasks"]) else "aggregate",
        {"retrieve": "retrieve", "aggregate": "aggregate"},
    )
    g.add_edge("aggregate", END)
    return g.compile()


def run_task(rt: Runtime, task: Task, engine: str = "auto", graph: Any = None) -> dict[str, Any]:
    """Run one task and return a flat result record."""
    rt.trace = TaskTrace()
    init: dict[str, Any] = {"task_id": task.id, "question": task.question}
    error = None
    try:
        if resolve_engine(engine) == "langgraph":
            graph = graph or build_langgraph(rt)
            limit = 4 * rt.cfg.agents.max_subtasks + 10
            state = graph.invoke(init, config={"recursion_limit": limit})
        else:
            state = run_native(rt, init)
        final = state.get("final", "")
    except Exception as exc:  # one failed task must not kill a multi-hour run
        final, error = "", f"{type(exc).__name__}: {exc}"
    latency = rt.trace.elapsed()
    success = bool(final) and score(final, task.answer)
    updates = rt.router.end_task(success)  # learned routers update here; heuristic is a no-op
    return {
        "task_id": task.id,
        "complexity": task.complexity,
        "source": task.source,
        "prediction": extract_final_answer(final),
        "gold": task.answer,
        "success": success,
        "routing_policy": rt.router.policy,
        "router_updates": updates,
        "latency_s": latency,
        "error": error,
        **rt.trace.summary(),
    }
