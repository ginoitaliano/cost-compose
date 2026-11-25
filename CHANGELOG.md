# Changelog

## Unreleased

Every new flag defaults to the previous behaviour; dry-run records for the existing configurations are unchanged on all previously recorded fields.

- `PREREGISTRATION.md` fixes the evaluation run: streams, 15 main and 9 reuse configurations, outcomes, a confirmatory set C1-C4 with Holm correction, and pre-declared estimands. The report leads with the confirmatory tests, adds bootstrap p-values, the iso-success oracle (with and without giving up on tasks baseline never solves), and input and output tokens per task. `make run-a`/`run-b`/`run-c` and `make merge` run the main stream as three processes. `costcompose report --exclude-tasks` writes a sensitivity report without given tasks (the pilot tasks). `composed_model_aware`, `composed_guarded` and `composed_structured` move to the extended set.
- `agents.max_output_tokens: 4096`: one output cap for every model's requests. Without it, vLLM let a degenerate Llama generation run toward the 131k context limit until the 120 s timeout.
- `serve_llama.sh`: Llama 3.1 at its native 131,072-token context (`MAX_MODEL_LEN`), was 8,192, which failed 3 of 60 Llama-only development runs; `--generation-config vllm`.
- Seed-averaged oracle bound next to the per-run one; routed configs are placed against it. `docs/DEVLOG.md`: every post-data change, what prompted it, and what it changed.
- Answer F1 (MuSiQue's official metric) on every record; mean F1 per config and paired F1 differences in the report.
- Prompts: reasoner, aggregator and single agent give a best answer instead of "unknown" and the answer rather than a sentence. This changes every configuration's requests; earlier runs are not comparable.
- Reference points: `llama_only`, `haiku_only`, `sonnet_only` (core). The report adds each single model's cost and success and the task-level oracle bound, with routed configs placed against it. `make run-dev-refs` runs them on the development set.
- Makefile targets set `LITELLM_LOG=ERROR` by default, and `-v` no longer turns on LiteLLM and httpx request logging (they stay at `LITELLM_LOG`, default WARNING).
- Structured cache: `caching.key: structured` embeds what a request asks and matches what its answer depends on exactly (role, retrieved document IDs, upstream answers, loop history, entities of the ask); `caching.subtask_cache: true` caches resolved subtask answers and skips the subtask on a hit. Defaults (`text`, `false`) leave every existing config unchanged. Core configs `cache_only_structured`, `composed_structured`, `cache_route_structured`; extended `cache_only_structured_keys`. The report adds "Subtask cache", `H1_structured_cache`, and paired comparisons of the structured cache against the guarded one. `make run-dev` runs the caches on the development set.
- Development set: `prepare_musique.py --exclude` and `make suite-dev` (20 per stratum, disjoint from both evaluation suites, own corpus). Method changes are made on it; the evaluation suites run once.
- Retrieval diagnostics: every MuSiQue task file gets `<out>.support.json` (supporting-paragraph document IDs per task and per hop, resolved sub-questions); run records gain `retrieved_doc_ids`; `scripts/retrieval_check.py ceiling | pipeline` reports recall@k with question and gold sub-question queries, and pipeline support recall with success split by it.
- Retrieval fix: a heading is never a chunk on its own. Long MuSiQue paragraphs were split into a title-only passage and an untitled passage. The sample corpus, and so every dry-run record, is unchanged.
- Reuse stream: `prepare_musique.py --reuse-share r --cluster-max k` draws a share of each stratum as clusters of MuSiQue questions that share a single-hop question; `make suite-reuse` and `make run-reuse` build and run it with r = 0.5 into separate files. `--reuse-share 0` (default) is unchanged. Every MuSiQue log now records `hop_sharing`. Prompted by the guarded-cache pilot: with false hits removed, unrelated questions gave the cache nothing to reuse.
- Guarded cache. New flags `caching.role_scoped`, `caching.first_step_only` and `caching.entity_guard`, all off by default, so the existing cache configs are unchanged and form the naive arm. Core configs `cache_only_guarded`, `composed_guarded` and `cache_route_guarded` turn all four guards on (with `skip_tool_calls`); extended configs turn one guard on at a time over `cache_only`. Prompted by the pilot: the aggregator was served the stored plan, and loop steps were served their own previous tool call.
- Hit-mechanism accounting: every cache entry records the task run, reasoning loop and role that wrote it; each hit is classified as same loop, same task or cross task, and flagged for cross-role and query-entity mismatch. The report adds "Cache hits by mechanism" (with success given a hit by origin) and `H1_guarded_cache`. Hit rates now divide by `cache_lookups`, which equals `requests` unless steps are bypassed.
- Routing scores: every record carries `[role, score]` per request; the report adds the score distribution per role against the ladder cutoffs. In the pilot no request reached the top rung's cutoff.
- H1 report wording: the sum-of-savings note now covers a negative saving, where the independence prediction is above the sum.
- Runs: a preflight call to every model and embedder before the first task (`--skip-preflight`); on resume, task runs that ended in an error move to `runs.errors.jsonl` and are retried (`--no-retry-errors`); tool calls are written one per assistant turn, since Llama 3.1's chat template rejects multi-call turns that a routed GPT-4o or Claude step can produce.
- Task suite: `scripts/prepare_musique.py` and `make suite`. GAIA leaves only 18 eligible tasks (123 of 165 need browsing), so MuSiQue (dev split, MuSiQue-Ans) fills each stratum to 40, stratified by hop count, with a shared paragraph corpus and a log carrying the input SHA-256. Tasks accept an optional `aliases` list; both scores accept any listed answer.
- `costcompose judge`: optional audit of the scorer by an LLM judge that must not be on the routing ladder; judges the runs where primary and strict disagree (and optionally the spot-check sample), and reports agreement with both scores. Never changes `runs.jsonl`. `LiteLLMClient` accepts an optional `api_base`.
- Scoring: every record has `success` (primary: quasi-exact match that also accepts trailing unit words the question names) and `success_strict` (GAIA-style quasi-exact match). The report adds a primary-vs-strict section with paired tests under both. The aggregator and single-agent prompts now ask for the bare number even when the question names a unit. The spot-check CSV carries both automated verdicts.
- Bottom routing rung is now Llama-3.1-8B-Instruct (`meta-llama/Llama-3.1-8B-Instruct`), replacing Llama-3-8B-Instruct; same size, same placeholder price. `serve_llama.sh` binds to 127.0.0.1 and one GPU by default.
- `costcompose spotcheck sample | score` and `make spotcheck`: seeded, stratified 10% sample of tasks for human grading (paper §3.4), and agreement with the automated scorer (raw agreement, Cohen's kappa, per config, with n).
- `make run` now runs an explicit list: Table 1 plus the ablations H1-H3 need. `make run-extended` runs the other ablations into the same output directory. A test checks that every ablation is in exactly one list.

- H1: the independence prediction `1 - (1 - s_cache)(1 - s_route)` is the primary test, with the interaction term (observed minus prediction) and a 95% paired bootstrap CI. The sum-of-savings comparison is kept as a labelled secondary figure.
- Cache correctness: regex entity extractor (`cache.extract_entities`); per-hit `entity_mismatch` flag; per config and scope, the hit rate, entity-mismatch share of hits and success given hit vs miss, each with n.
- Composition effect 4: `cache_induced_early_stops`, `cached_drafts` and `cached_draft_replays` per task; new flag `early_stopping.exclude_cached_drafts`; ablation `composed_exclude_cached_drafts`.
- `caching.skip_tool_calls`: never store or serve responses with tool-call arguments; ablation `composed_skip_tool_calls`; per-task `tool_call_responses_not_cached`.
- `caching.version_key`: instruction version in cache namespaces for both scopes (`prompts`, `file:<path>` or a literal).
- Statistics: paired comparisons over tasks lead the report. They use an exact McNemar test on (task, seed) pairs, and paired bootstrap CIs (10k resamples, seed 0) for success, cost and latency differences. H3 per-stratum drops have stratified bootstrap CIs and a high-minus-low difference-of-differences. Every rate prints its n. Seed-level mean ± std moves to per-config descriptives.
- `mode: single_agent`: one reasoning agent with retrieval and tools as functions, on both engines. New ablations `single_agent_baseline` and `single_agent_route`. H3 reports multi-agent vs single-agent drops and their difference.
- `scripts/prepare_gaia.py`: validation split only. Excludes tasks needing web browsing, unreadable attachments or multimodal input, from GAIA metadata, and logs counts by reason to `<out>.exclusions.json`.
- Docs: repository URLs now point at `ginoitaliano/cost-compose`; added a package/CLI naming note and a complete `.env.example`. The cache is described as GPTCache-style and early stopping as a loop-level rule inspired by CALM. The paper map lists where the paper text and the code differ. `PR_TRIAGE.md` covers the open Dependabot PRs.
- Report and record changes: `summary.json` gains `comparisons`; `H1_sub_additive` and `H3_routing_success_drop` change shape (see README); run records gain the fields listed in the README.

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
