# Paper

`Altarouti_WiML_NeurIPS2026.docx`: Composing Cost-Reduction Techniques in Multi-Agent LLM Systems: A Practitioner's Empirical Study. NeurIPS 2026 Workshop on Women in Machine Learning (WiML).

## Where the paper maps to code

| Paper | Code |
| --- | --- |
| §3.1 pipeline (retrieval agent, reasoning agent, orchestrator) | `agents.py`, `pipeline.py` |
| Table 1 configurations | `configs/{baseline,cache_only,route_only,composed}.yaml` |
| §3.2 semantic caching, cosine 0.92, agent/pipeline scope | `cache.py`, `caching:` in `configs/base.yaml` |
| §3.2 routing, four-feature complexity heuristic | `routing.py`, `routing:` in `configs/base.yaml` |
| §3.2 early stopping, threshold 0.88 | `early_stop.py` |
| §3.3 task suite, 40 per stratum | `tasks.py`, `scripts/prepare_gaia.py` |
| §3.4 metrics (cost, P50/P95 latency, success, cache hit rate, routing distribution) | `tracking.py`, `analysis.py` |
| §3.5 three seeds, mean and std | `experiment.py`, `analysis.py` |
| §4 H1, H2, H3 | `analysis.hypotheses()` |
| §5 "adaptive composition policies" (future work) | `routing.LinUCBRouter`, `configs/ablations/learned_route.yaml` |
