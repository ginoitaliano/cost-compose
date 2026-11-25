"""Per-call accounting: tokens, USD cost, cache and routing outcomes."""

from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field

from .config import ModelPrice


def price_usd(prices: dict[str, ModelPrice], model: str, prompt: int, completion: int) -> float:
    p = prices[model]
    return (prompt * p.input_per_mtok + completion * p.output_per_mtok) / 1_000_000


@dataclass
class CallRecord:
    agent: str
    model: str  # model the router chose
    tier: str
    complexity_score: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cost_reference_usd: float  # same tokens priced at the reference model (GPT-4o)
    cache_hit: bool
    cross_model_hit: bool
    embedding_tokens: int
    latency_s: float
    entity_mismatch: bool = False  # cache hit served from a request with different entities
    role: str = ""  # the agent's role for this request: planner, retriever, reasoner, ...
    cache_lookup: bool = True  # False when the cache was bypassed (first_step_only)
    hit_origin: str = ""  # on a hit: "same_loop" | "same_task" | "cross_task" | "unknown"
    cross_role_hit: bool = False  # hit served from an entry written by another role
    query_entity_mismatch: bool = False  # hit whose query-text entities differ
    guard_refused: bool = False  # a hit the entity guard refused (served by the model instead)
    model_cost_usd: float = 0.0  # the provider call alone, without cache-lookup embeddings


def _by_model(calls: list[CallRecord]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for c in calls:
        row = out.setdefault(c.model, [0, 0, 0.0])
        row[0] += c.prompt_tokens
        row[1] += c.completion_tokens
        row[2] += c.model_cost_usd
    return out


@dataclass
class TaskTrace:
    """Everything recorded while one task runs through the pipeline."""

    calls: list[CallRecord] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)  # tags cache entries
    loops: int = 0  # reasoning loops started; the current one tags cache entries
    retrieved: list = field(default_factory=list)  # doc IDs returned by each search, in order
    subtask_cache_lookups: int = 0
    subtask_cache_hits: int = 0
    subtask_hits_by_origin: Counter = field(default_factory=Counter)
    reasoning_iterations: int = 0
    early_stops: int = 0
    cached_drafts: int = 0  # reasoner drafts served from the cache
    cached_draft_replays: int = 0  # ...whose answer repeats an earlier draft in the same loop
    cache_induced_early_stops: int = 0  # stops that would not have fired without cached drafts
    tool_call_responses_not_cached: int = 0  # withheld from the cache by skip_tool_calls
    subtasks: int = 0
    tool_invocations: int = 0
    aux_cost_usd: float = 0.0  # retrieval-query embeddings, not tied to an LLM request
    started: float = field(default_factory=time.perf_counter)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def summary(self) -> dict:
        llm_calls = [c for c in self.calls if not c.cache_hit]
        hits = [c for c in self.calls if c.cache_hit]
        lookups = [c for c in self.calls if c.cache_lookup]
        by_agent_lookups: Counter = Counter(c.agent for c in self.calls)
        by_agent_hits: Counter = Counter(c.agent for c in hits)
        return {
            "cost_usd": sum(c.cost_usd for c in self.calls) + self.aux_cost_usd,
            "cost_reference_usd": sum(c.cost_reference_usd for c in self.calls)
            + self.aux_cost_usd,
            "prompt_tokens": sum(c.prompt_tokens for c in llm_calls),
            "completion_tokens": sum(c.completion_tokens for c in llm_calls),
            "llm_calls": len(llm_calls),
            "requests": len(self.calls),
            "cache_hits": len(hits),
            "cache_lookups": len(lookups),
            "cache_bypassed": len(self.calls) - len(lookups),
            "hits_by_origin": dict(Counter(c.hit_origin for c in hits)),
            "hits_by_role": dict(Counter(c.role for c in hits)),
            "lookups_by_role": dict(Counter(c.role for c in lookups)),
            "cross_role_hits": sum(c.cross_role_hit for c in hits),
            "query_entity_mismatch_hits": sum(c.query_entity_mismatch for c in hits),
            "entity_guard_refusals": sum(c.guard_refused for c in self.calls),
            # (role, routing score) per request, for the score distribution against rung cutoffs
            "retrieved_doc_ids": [list(ids) for ids in self.retrieved],
            "subtask_cache_lookups": self.subtask_cache_lookups,
            "subtask_cache_hits": self.subtask_cache_hits,
            "subtask_hits_by_origin": dict(self.subtask_hits_by_origin),
            "routing_scores": [[c.role or c.agent, round(c.complexity_score, 4)]
                               for c in self.calls],
            "cross_model_hits": sum(c.cross_model_hit for c in hits),
            "entity_mismatch_hits": sum(c.entity_mismatch for c in hits),
            "entity_mismatch_hits_by_agent": dict(
                Counter(c.agent for c in hits if c.entity_mismatch)
            ),
            "cache_hits_by_agent": dict(by_agent_hits),
            "requests_by_agent": dict(by_agent_lookups),
            "routing": dict(Counter(c.model for c in self.calls)),
            # per model: [prompt tokens, completion tokens, USD] of provider calls, for repricing
            "tokens_by_model": _by_model(llm_calls),
            "routing_tiers": dict(Counter(c.tier for c in self.calls)),
            "reasoning_iterations": self.reasoning_iterations,
            "early_stops": self.early_stops,
            "cached_drafts": self.cached_drafts,
            "cached_draft_replays": self.cached_draft_replays,
            "cache_induced_early_stops": self.cache_induced_early_stops,
            "tool_call_responses_not_cached": self.tool_call_responses_not_cached,
            "subtasks": self.subtasks,
            "tool_invocations": self.tool_invocations,
        }
