"""Agent nodes and the shared runtime that routes, caches and meters every LLM request.

Nodes are plain functions ``node(runtime, state) -> partial state``. Both the native engine and
the LangGraph engine call the same functions, so results do not depend on the engine.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .cache import SemanticCache, dependency_key, extract_entities
from .config import ExperimentConfig
from .early_stop import EarlyStopper, extract_answer
from .llm import LLMClient, LLMResponse, approx_tokens
from .retrieval import VectorStore
from .routing import ComplexityFeatures, Router
from .tools import ToolRegistry
from .tracking import CallRecord, TaskTrace, price_usd


@dataclass
class Runtime:
    cfg: ExperimentConfig
    llm: LLMClient
    router: Router
    stopper: EarlyStopper
    store: VectorStore
    cache: SemanticCache | None = None
    seed: int | None = None
    trace: TaskTrace = field(default_factory=TaskTrace)
    tools: ToolRegistry = field(init=False)
    current_loop: int | None = None  # id of the reasoning loop running now, within the task
    pending_subtask: Any = None  # (subtask, embedding) awaiting its answer for the subtask cache

    def __post_init__(self) -> None:
        self.tools = ToolRegistry(search_fn=self._search_text)

    # -- metered helpers ---------------------------------------------------------------------

    def _embed_cost(self, tokens: int) -> float:
        return price_usd(self.cfg.prices, self.cfg.embedding_model, tokens, 0)

    def _search_text(self, query: str) -> str:
        passages, tokens = self.store.search(query, k=self.cfg.agents.top_k)
        self.trace.aux_cost_usd += self._embed_cost(tokens)
        self.trace.retrieved.append([p.doc_id for p in passages])
        return "\n\n".join(f"[{p.doc_id}] {p.text}" for p in passages) or "(no documents)"

    def call_llm(
        self,
        agent: str,
        messages: list[dict[str, Any]],
        features: ComplexityFeatures,
        local_input: str,
        pipeline_context: str,
        use_tools: bool = False,
        role: str | None = None,
        entity_text: str | None = None,
        loop_step: int = 0,
        cache_query: str | None = None,
        cache_deps: tuple = (),
    ) -> LLMResponse:
        """The single choke point for LLM requests: route, look up the cache, call, meter.

        ``role`` is the agent's role for this request (the orchestrator plans and aggregates).
        ``entity_text`` is the query text for the cache's entity checks. ``loop_step`` is the
        request's position in a reasoning loop (0 outside loops), for ``first_step_only``.
        ``cache_query`` and ``cache_deps`` are used only with ``caching.key: structured``: the
        text that is embedded (what is asked) and what the answer depends on, matched exactly
        together with the entities of the ask.
        """
        role = role or agent
        deps = None
        if self.cfg.caching.key == "structured":
            local_input = cache_query if cache_query is not None else local_input
            ask = entity_text if entity_text is not None else local_input
            deps = dependency_key(role, list(cache_deps), sorted(extract_entities(ask)))
        decision = self.router.route(features)
        t0 = time.perf_counter()
        vec, emb_tokens = None, 0
        use_cache = self.cache is not None and not (
            self.cfg.caching.first_step_only and loop_step > 0
        )
        refused = False
        if use_cache:
            before = self.cache.stats.entity_guard_refusals[agent]
            hit, vec, emb_tokens = self.cache.lookup(
                agent, decision.model, local_input, pipeline_context, role, entity_text, deps
            )
            refused = self.cache.stats.entity_guard_refusals[agent] > before
            if hit is not None:
                emb = self._embed_cost(emb_tokens)
                origin = self._hit_origin(hit.origin)
                cross_role = bool(hit.origin) and hit.origin.get("role", role) != role
                self.trace.calls.append(
                    CallRecord(agent, decision.model, decision.tier, decision.score, 0, 0, emb,
                               emb, True, hit.source_model != decision.model, emb_tokens,
                               time.perf_counter() - t0, hit.entity_mismatch, role, True,
                               origin, cross_role, hit.query_entity_mismatch)
                )
                return hit.response

        resp = self.llm.complete(
            decision.model,
            messages,
            tools=self.tools.schemas if use_tools else None,
            temperature=self.cfg.agents.temperature,
            seed=self.seed,
        )
        if use_cache and vec is not None:
            origin = {"run": self.trace.run_id, "loop": self.current_loop, "role": role}
            if not self.cache.store(agent, decision.model, vec, resp, local_input, role,
                                    entity_text, origin, deps):
                self.trace.tool_call_responses_not_cached += 1
        u = resp.usage
        emb = self._embed_cost(emb_tokens)
        cost = price_usd(self.cfg.prices, decision.model, u.prompt_tokens, u.completion_tokens)
        ref = price_usd(self.cfg.prices, self.cfg.reference_model, u.prompt_tokens,
                        u.completion_tokens)
        self.router.observe(decision, cost, ref)
        self.trace.calls.append(
            CallRecord(agent, decision.model, decision.tier, decision.score, u.prompt_tokens,
                       u.completion_tokens, cost + emb, ref + emb, False, False, emb_tokens,
                       time.perf_counter() - t0, role=role, cache_lookup=use_cache,
                       guard_refused=refused)
        )
        return resp

    # -- subtask cache ---------------------------------------------------------------------

    def _subtask_deps(self, state: dict[str, Any], sub: str) -> str:
        """A subtask's answer depends on the answers before it and the entities it names."""
        return dependency_key("subtask", list(state.get("outputs", [])),
                              sorted(extract_entities(sub)))

    def subtask_lookup(self, state: dict[str, Any], sub: str) -> tuple[str | None, Any]:
        """Look up a resolved answer for this subtask. Returns (answer or None, embedding)."""
        self.trace.subtask_cache_lookups += 1
        hit, vec, tokens = self.cache.lookup("subtask", "*", sub, "", "subtask", sub,
                                             self._subtask_deps(state, sub))
        self.trace.aux_cost_usd += self._embed_cost(tokens)
        if hit is None:
            return None, vec
        self.trace.subtask_cache_hits += 1
        self.trace.subtask_hits_by_origin[self._hit_origin(hit.origin)] += 1
        return hit.response.content, vec

    def subtask_store(self, state: dict[str, Any], sub: str, vec: Any, answer: str) -> None:
        origin = {"run": self.trace.run_id, "loop": None, "role": "subtask"}
        self.cache.store("subtask", "*", vec, LLMResponse(answer, "subtask-cache"), sub,
                         "subtask", sub, origin, self._subtask_deps(state, sub))

    def _hit_origin(self, origin: dict | None) -> str:
        """Where a served entry was written, relative to the request it now answers."""
        if not origin:
            return "unknown"
        if origin.get("run") != self.trace.run_id:
            return "cross_task"
        if self.current_loop is not None and origin.get("loop") == self.current_loop:
            return "same_loop"
        return "same_task"


# -- helpers ---------------------------------------------------------------------------------


def parse_plan(text: str, fallback: str, max_subtasks: int) -> list[str]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            items = json.loads(m.group(0))
            steps = [str(s).strip() for s in items if str(s).strip()]
            if steps:
                return steps[:max_subtasks]
        except json.JSONDecodeError:
            pass
    return [fallback]


def _upstream(state: dict[str, Any]) -> str:
    outs = state.get("outputs", [])
    subs = state.get("subtasks", [])
    return "\n".join(f"- {subs[i]} -> {o}" for i, o in enumerate(outs)) or "(none)"


def _pipeline_context(state: dict[str, Any]) -> str:
    return f"Task: {state['question']}\nEarlier results:\n{_upstream(state)}"


# -- nodes -----------------------------------------------------------------------------------


def plan_node(rt: Runtime, state: dict[str, Any]) -> dict[str, Any]:
    q = state["question"]
    system = prompts.PLANNER.format(max_subtasks=rt.cfg.agents.max_subtasks)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": q}]
    feats = ComplexityFeatures(approx_tokens(q), 0, 0, 0)
    resp = rt.call_llm("orchestrator", messages, feats, q, "", role="planner", entity_text=q,
                       cache_query=q)
    subtasks = parse_plan(resp.content, q, rt.cfg.agents.max_subtasks)
    rt.trace.subtasks = len(subtasks)
    return {"subtasks": subtasks, "index": 0, "outputs": [], "context": ""}


def retrieve_node(rt: Runtime, state: dict[str, Any]) -> dict[str, Any]:
    sub = state["subtasks"][state["index"]]
    if rt.cfg.caching.subtask_cache and rt.cache is not None:
        answer, vec = rt.subtask_lookup(state, sub)
        rt.pending_subtask = None if answer is not None else (sub, vec)
        if answer is not None:  # resolved before: skip retrieval and reasoning
            return {"context": "", "context_docs": [], "cached_answer": answer}
    passages, tokens = rt.store.search(sub, k=rt.cfg.agents.top_k)
    rt.trace.aux_cost_usd += rt._embed_cost(tokens)
    docs = [p.doc_id for p in passages]
    rt.trace.retrieved.append(docs)
    ptext = "\n\n".join(f"[{p.doc_id}] {p.text}" for p in passages) or "(no documents)"
    user = f"Subtask: {sub}\n\nPassages:\n{ptext}"
    messages = [
        {"role": "system", "content": prompts.RETRIEVER},
        {"role": "user", "content": user},
    ]
    feats = ComplexityFeatures(approx_tokens(sub), 0, approx_tokens(ptext), state["index"] + 1)
    resp = rt.call_llm("retriever", messages, feats, user, _pipeline_context(state),
                       role="retriever", entity_text=sub, cache_query=sub,
                       cache_deps=(sorted(docs),))
    return {"context": resp.content, "context_docs": docs, "cached_answer": None}


def _reasoning_loop(
    rt: Runtime,
    agent: str,
    system: str,
    user: str,
    query: str,
    depth: int,
    pipeline_context: str,
    deps: tuple = (),
) -> list[str]:
    """Draft / tool-call / refine loop with early stopping. Returns the drafts produced.

    ``deps`` is what the loop's answer depends on besides the ask itself (structured cache keys);
    each step adds the loop's transcript so far, so a step can only be served a response
    produced after exactly the same history.

    Shared by the multi-agent reasoner and the single agent so both run the same stop rule,
    tool handling and cache instrumentation.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    drafts: list[str] = []
    cached: list[bool] = []  # per draft: served from the cache rather than generated
    transcript: list[str] = []
    tools_used = 0
    rt.trace.loops += 1
    rt.current_loop = rt.trace.loops  # tags cache entries and classifies hits as same-loop
    for step in range(rt.cfg.agents.max_iterations):
        context_tokens = sum(approx_tokens(str(m.get("content") or "")) for m in messages)
        feats = ComplexityFeatures(approx_tokens(query), tools_used, context_tokens, depth)
        local = user + "\n" + "\n".join(transcript)
        resp = rt.call_llm(agent, messages, feats, local, pipeline_context, use_tools=True,
                           role=agent, entity_text=query, loop_step=step, cache_query=query,
                           cache_deps=(*deps, list(transcript)))
        if resp.tool_calls:
            # History holds one tool call per assistant turn, each followed by its result, even
            # when a model requested several at once. Every provider accepts this form, while
            # some chat templates (Llama 3.1) reject several calls in one turn; with routing,
            # the next step of this loop may go to such a model.
            for i, tc in enumerate(resp.tool_calls):
                messages.append({
                    "role": "assistant",
                    "content": (resp.content or None) if i == 0 else None,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    ],
                })
                result = rt.tools.call(tc.name, tc.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                transcript.append(f"tool {tc.name}({tc.arguments}) -> {result}")
            tools_used += len(resp.tool_calls)
            rt.trace.tool_invocations += len(resp.tool_calls)
            continue

        from_cache = bool(rt.trace.calls) and rt.trace.calls[-1].cache_hit
        if from_cache:
            rt.trace.cached_drafts += 1
            answer = extract_answer(resp.content)
            if any(extract_answer(d) == answer for d in drafts):
                rt.trace.cached_draft_replays += 1
        drafts.append(resp.content)
        cached.append(from_cache)
        transcript.append(resp.content)
        rt.trace.reasoning_iterations += 1
        stop_all, _conf = rt.stopper.should_stop(drafts)
        fresh = [d for d, c in zip(drafts, cached, strict=True) if not c]
        stop_fresh, _ = rt.stopper.should_stop(fresh)
        stop = stop_fresh if rt.cfg.early_stopping.exclude_cached_drafts else stop_all
        if stop:
            rt.trace.early_stops += 1
            if not stop_fresh:  # the cached drafts are what made the loop look converged
                rt.trace.cache_induced_early_stops += 1
            break
        if re.search(r"STATUS:\s*FINAL", resp.content, re.IGNORECASE):
            break
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": prompts.REFINE})
    rt.current_loop = None
    return drafts


def reason_node(rt: Runtime, state: dict[str, Any]) -> dict[str, Any]:
    idx = state["index"]
    sub = state["subtasks"][idx]
    if state.get("cached_answer") is not None:  # the subtask cache resolved it
        return {"outputs": [*state["outputs"], state["cached_answer"]], "index": idx + 1,
                "cached_answer": None}
    user = (
        f"Task: {state['question']}\nSubtask: {sub}\n\nContext:\n{state['context']}"
        f"\n\nEarlier results:\n{_upstream(state)}"
    )
    upstream = [[s, o] for s, o in zip(state["subtasks"], state["outputs"], strict=False)]
    drafts = _reasoning_loop(rt, "reasoner", prompts.REASONER, user, sub, idx + 1,
                             _pipeline_context(state),
                             deps=(sorted(state.get("context_docs", [])), upstream))
    answer = extract_answer(drafts[-1]) if drafts else ""
    if rt.pending_subtask is not None and answer:
        pending_sub, vec = rt.pending_subtask
        if pending_sub == sub:
            rt.subtask_store(state, sub, vec, answer)
    rt.pending_subtask = None
    return {"outputs": [*state["outputs"], answer], "index": idx + 1}


def single_agent_node(rt: Runtime, state: dict[str, Any]) -> dict[str, Any]:
    """H3 comparison arm: one reasoning agent, retrieval and tools as functions, no orchestrator.

    Every LLM request still goes through ``Runtime.call_llm``, so caching, routing, early
    stopping and metering behave exactly as in the multi-agent pipeline. There is no upstream
    agent, so the dependency-depth routing feature is 0 for every call.
    """
    q = state["question"]
    user = f"Task: {q}"
    rt.trace.subtasks = 1
    drafts = _reasoning_loop(rt, "single_agent", prompts.SINGLE_AGENT, user, q, 0, user)
    answer = extract_answer(drafts[-1]) if drafts else ""
    return {"subtasks": [q], "index": 1, "outputs": [answer], "context": "",
            "final": f"FINAL ANSWER: {answer}" if answer else ""}


def aggregate_node(rt: Runtime, state: dict[str, Any]) -> dict[str, Any]:
    results = _upstream(state)
    user = f"Task: {state['question']}\n\nSubtask results:\n{results}"
    messages = [
        {"role": "system", "content": prompts.AGGREGATOR},
        {"role": "user", "content": user},
    ]
    depth = len(state["subtasks"]) + 1
    feats = ComplexityFeatures(approx_tokens(state["question"]), 0, approx_tokens(results), depth)
    resp = rt.call_llm("orchestrator", messages, feats, user, f"Task: {state['question']}",
                       role="aggregator", entity_text=state["question"],
                       cache_query=state["question"], cache_deps=(results,))
    return {"final": resp.content}
