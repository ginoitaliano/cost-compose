# Changelog

## 0.2.0

- Learned routing: `routing.policy: linucb`, a disjoint LinUCB contextual bandit over the four complexity features, trained online on task success minus normalised cost. Cache hits are excluded from updates.
- New ablations `learned_route` and `composed_learned`.
- Report adds per-quarter learning curves and a learned-vs-heuristic comparison; final policy state saved to `policies/`.
- Records include stream position, routing policy and router update count.

## 0.1.0

- Reference pipeline (orchestrator, retrieval agent, reasoning agent) on LangGraph, with a dependency-free native engine that runs the same nodes.
- Semantic cache with agent and pipeline scopes, model-aware mode and cross-model hit accounting; in-memory and Redis backends.
- Four-feature complexity router over a LiteLLM model ladder (Llama-3-8B, Claude Haiku 4.5, GPT-4o, Claude Sonnet 4.5).
- Confidence-based early stopping (self-reported confidence + draft consistency).
- Resumable experiment runner, per-seed statistics, H1–H3 hypothesis checks, Markdown and JSON reports.
- Offline fake backend, 38 tests, CI on Python 3.10–3.12 with a Redis + LangGraph job, Docker image.
