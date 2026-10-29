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
from .cache import SemanticCache
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

    def __post_init__(self) -> None:
        self.tools = ToolRegistry(search_fn=self._search_text)

    # -- metered helpers ---------------------------------------------------------------------

    def _embed_cost(self, tokens: int) -> float:
        return price_usd(self.cfg.prices, self.cfg.embedding_model, tokens, 0)

    def _search_text(self, query: str) -> str:
        passages, tokens = self.store.search(query, k=self.cfg.agents.top_k)
        self.trace.aux_cost_usd += self._embed_cost(tokens)
        return "\n\n".join(f"[{p.doc_id}] {p.text}" for p in passages) or "(no documents)"

    def call_llm(
        self,
        agent: str,
        messages: list[dict[str, Any]],
        features: ComplexityFeatures,
        local_input: str,
        pipeline_context: str,
        use_tools: bool = False,
    ) -> LLMResponse:
        decision = self.router.route(features)
        t0 = time.perf_counter()
        vec, emb_tokens = None, 0
        if self.cache is not None:
            hit, vec, emb_tokens = self.cache.lookup(
                agent, decision.model, local_input, pipeline_context
            )
            if hit is not None:
                emb = self._embed_cost(emb_tokens)
                self.trace.calls.append(
                    CallRecord(agent, decision.model, decision.tier, decision.score, 0, 0, emb,
                               emb, True, hit.source_model != decision.model, emb_tokens,
                               time.perf_counter() - t0)
                )
                return hit.response

        resp = self.llm.complete(
            decision.model,
            messages,
            tools=self.tools.schemas if use_tools else None,
            temperature=self.cfg.agents.temperature,
            seed=self.seed,
        )
        if self.cache is not None and vec is not None:
            self.cache.store(agent, decision.model, vec, resp)
        u = resp.usage
        emb = self._embed_cost(emb_tokens)
        cost = price_usd(self.cfg.prices, decision.model, u.prompt_tokens, u.completion_tokens)
        ref = price_usd(self.cfg.prices, self.cfg.reference_model, u.prompt_tokens,
                        u.completion_tokens)
        self.router.observe(decision, cost, ref)
        self.trace.calls.append(
            CallRecord(agent, decision.model, decision.tier, decision.score, u.prompt_tokens,
                       u.completion_tokens, cost + emb, ref + emb, False, False, emb_tokens,
                       time.perf_counter() - t0)
        )
        return resp


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
    resp = rt.call_llm("orchestrator", messages, feats, q, "")
    subtasks = parse_plan(resp.content, q, rt.cfg.agents.max_subtasks)
    rt.trace.subtasks = len(subtasks)
    return {"subtasks": subtasks, "index": 0, "outputs": [], "context": ""}


def retrieve_node(rt: Runtime, state: dict[str, Any]) -> dict[str, Any]:
    sub = state["subtasks"][state["index"]]
    passages, tokens = rt.store.search(sub, k=rt.cfg.agents.top_k)
    rt.trace.aux_cost_usd += rt._embed_cost(tokens)
    ptext = "\n\n".join(f"[{p.doc_id}] {p.text}" for p in passages) or "(no documents)"
    user = f"Subtask: {sub}\n\nPassages:\n{ptext}"
    messages = [
        {"role": "system", "content": prompts.RETRIEVER},
        {"role": "user", "content": user},
    ]
    feats = ComplexityFeatures(approx_tokens(sub), 0, approx_tokens(ptext), state["index"] + 1)
    resp = rt.call_llm("retriever", messages, feats, user, _pipeline_context(state))
    return {"context": resp.content}


def reason_node(rt: Runtime, state: dict[str, Any]) -> dict[str, Any]:
    idx = state["index"]
    sub = state["subtasks"][idx]
    user = (
        f"Task: {state['question']}\nSubtask: {sub}\n\nContext:\n{state['context']}"
        f"\n\nEarlier results:\n{_upstream(state)}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts.REASONER},
        {"role": "user", "content": user},
    ]
    drafts: list[str] = []
    transcript: list[str] = []
    tools_used = 0
    for _ in range(rt.cfg.agents.max_iterations):
        context_tokens = sum(approx_tokens(str(m.get("content") or "")) for m in messages)
        feats = ComplexityFeatures(approx_tokens(sub), tools_used, context_tokens, idx + 1)
        local = user + "\n" + "\n".join(transcript)
        resp = rt.call_llm("reasoner", messages, feats, local, _pipeline_context(state),
                           use_tools=True)
        if resp.tool_calls:
            messages.append({
                "role": "assistant",
                "content": resp.content or None,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in resp.tool_calls
                ],
            })
            for tc in resp.tool_calls:
                result = rt.tools.call(tc.name, tc.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                transcript.append(f"tool {tc.name}({tc.arguments}) -> {result}")
            tools_used += len(resp.tool_calls)
            rt.trace.tool_invocations += len(resp.tool_calls)
            continue

        drafts.append(resp.content)
        transcript.append(resp.content)
        rt.trace.reasoning_iterations += 1
        stop, _conf = rt.stopper.should_stop(drafts)
        if stop:
            rt.trace.early_stops += 1
            break
        if re.search(r"STATUS:\s*FINAL", resp.content, re.IGNORECASE):
            break
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": prompts.REFINE})

    answer = extract_answer(drafts[-1]) if drafts else ""
    return {"outputs": [*state["outputs"], answer], "index": idx + 1}


def aggregate_node(rt: Runtime, state: dict[str, Any]) -> dict[str, Any]:
    results = _upstream(state)
    user = f"Task: {state['question']}\n\nSubtask results:\n{results}"
    messages = [
        {"role": "system", "content": prompts.AGGREGATOR},
        {"role": "user", "content": user},
    ]
    depth = len(state["subtasks"]) + 1
    feats = ComplexityFeatures(approx_tokens(state["question"]), 0, approx_tokens(results), depth)
    resp = rt.call_llm("orchestrator", messages, feats, user, f"Task: {state['question']}")
    return {"final": resp.content}
