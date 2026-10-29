"""Per-call accounting: tokens, USD cost, cache and routing outcomes."""

from __future__ import annotations

import time
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


@dataclass
class TaskTrace:
    """Everything recorded while one task runs through the pipeline."""

    calls: list[CallRecord] = field(default_factory=list)
    reasoning_iterations: int = 0
    early_stops: int = 0
    subtasks: int = 0
    tool_invocations: int = 0
    aux_cost_usd: float = 0.0  # retrieval-query embeddings, not tied to an LLM request
    started: float = field(default_factory=time.perf_counter)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def summary(self) -> dict:
        llm_calls = [c for c in self.calls if not c.cache_hit]
        hits = [c for c in self.calls if c.cache_hit]
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
            "cross_model_hits": sum(c.cross_model_hit for c in hits),
            "cache_hits_by_agent": dict(by_agent_hits),
            "requests_by_agent": dict(by_agent_lookups),
            "routing": dict(Counter(c.model for c in self.calls)),
            "routing_tiers": dict(Counter(c.tier for c in self.calls)),
            "reasoning_iterations": self.reasoning_iterations,
            "early_stops": self.early_stops,
            "subtasks": self.subtasks,
            "tool_invocations": self.tool_invocations,
        }
