# Paper

`Altarouti_WiML_NeurIPS2026.docx`: Composing Cost-Reduction Techniques in Multi-Agent LLM Systems: A Practitioner's Empirical Study. NeurIPS 2026 Workshop on Women in Machine Learning (WiML).

## Where the paper maps to code

| Paper | Code |
| --- | --- |
| §3.1 pipeline (retrieval agent, reasoning agent, orchestrator) | `agents.py`, `pipeline.py` (`mode: multi_agent`) |
| Table 1 configurations | `configs/{baseline,cache_only,route_only,composed}.yaml` |
| §3.2 semantic caching, cosine 0.92, `text-embedding-3-small`, agent/pipeline scope | `cache.py`, `caching:` in `configs/base.yaml`. GPTCache-style, reimplemented for per-scope keys and cross-model hit accounting; the GPTCache library is not used. |
| §3.2 routing, four-feature complexity heuristic, LiteLLM model ladder | `routing.py`, `routing:` in `configs/base.yaml` |
| §3.2 early stopping, threshold 0.88 | `early_stop.py`. A loop-level stop rule inspired by CALM (Schuster et al. 2022), not CALM: CALM is token-level early exit, while this rule decides whether to run another reasoning-loop iteration. |
| §3.3 task suite, 40 per stratum | `tasks.py`, `scripts/prepare_gaia.py` (GAIA validation split; excludes web-browsing, attachment and multimodal tasks, with counts by reason) |
| §3.4 metrics (cost, P50/P95 latency, success, cache hit rate, routing distribution) | `tracking.py`, `analysis.py` |
| §3.4 task success (automated evaluation against ground truth) | `evaluation.score_lenient` (primary) and `evaluation.score` (strict, GAIA-style); both recorded per task |
| §3.4 human spot-checks on 10% of tasks | `spotcheck.py`, `costcompose spotcheck sample` / `score`, `make spotcheck` |
| §3.5 three seeds, means and standard deviations | `experiment.py`; `analysis.py` reports them as per-config descriptives, with paired tests over tasks as the headline comparison |
| §4 H1 | `analysis._h1_test`: independence baseline `1 - (1 - s_cache)(1 - s_route)` as the primary test, interaction term with a 95% CI; the paper's sum-of-savings framing as a secondary figure |
| §4 H2 | `analysis.hypotheses` (`H2_pipeline_vs_agent_cache`), `configs/ablations/cache_only_pipeline.yaml`; correctness side in `cache_correctness` (entity-mismatch hits) |
| §4 H3 | `analysis._h3_test`: per-stratum routing drop, high-minus-low difference-of-differences, and a measured single-agent comparison (`configs/ablations/single_agent_*.yaml`) in place of single-agent figures from the literature |
| §5 "adaptive composition policies" (future work) | `routing.LinUCBRouter`, `configs/ablations/learned_route.yaml` |

## In the code, not in the paper

| Addition | Code |
| --- | --- |
| Composition effect 4: early stops triggered by cache-replayed drafts | `cache_induced_early_stops`, `configs/ablations/composed_exclude_cached_drafts.yaml` |
| Tool-call caching guard | `caching.skip_tool_calls`, `configs/ablations/composed_skip_tool_calls.yaml` |
| Instruction versioning in cache keys | `caching.version_key` |
| Paired statistics (McNemar, paired bootstrap) | `analysis.compare` |
| Primary (unit-tolerant) and strict scores, both reported | `evaluation.py`, "Scoring: primary vs strict" in the report |
| Optional judge audit of the scorer by an off-ladder model | `judge.py`, `costcompose judge` |

## Where the paper text and the code differ

These are not changed in the code. They are listed so the paper text can be brought in line.

- The proposal's ladder starts at Llama-3-8B. The code uses Llama-3.1-8B-Instruct, the current 8B instruct model in the Llama 3 line; the other three rungs are unchanged. The paper should name the model actually run.

- §3.2 says caching is implemented "using GPTCache". The code reimplements GPTCache's design (see above).
- §3.2 cites CALM for early stopping "in the spirit of confidence-based adaptive computation". The code's rule is loop-level, not token-level.
- §3.4 says metrics are collected "via LangSmith tracing". The code meters tokens, cost and latency itself in `tracking.py` for both engines. LangSmith traces only the LangGraph engine and is not the source of any reported number.
- §3.4 says cost per task is normalised to GPT-4o pricing. The headline cost (`cost_usd`), and the savings in H1, price each call at the model that ran it. The GPT-4o-priced figure is recorded separately as `cost_reference_usd`.
- §3.5 reports means and standard deviations across seeds. The code still reports those, but leads with paired tests over tasks.
- §4 H1 is stated against the sum of individual savings. The code treats that comparison as secondary because any two positive savings compose below their sum.
- §4 H3 compares against single-agent results "reported" in the literature. The code measures a single-agent arm instead.
