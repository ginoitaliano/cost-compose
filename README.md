# Composing Cost-Reduction Techniques in Multi-Agent LLM Systems

[![CI](https://github.com/ginoitaliano/cost-compose/actions/workflows/ci.yml/badge.svg)](https://github.com/ginoitaliano/cost-compose/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Repository description: Harness for measuring how semantic caching, model routing and early stopping compose in a multi-agent LLM pipeline (NeurIPS 2026 WiML).
Suggested GitHub topics: `llm`, `multi-agent`, `llm-inference`, `semantic-caching`, `model-routing`, `early-stopping`, `contextual-bandit`, `langgraph`, `benchmark`.

Code for *Composing Cost-Reduction Techniques in Multi-Agent LLM Systems: A Practitioner's Empirical Study* (Sarah Ahmed Altarouti, NeurIPS 2026 Workshop on Women in Machine Learning).

Semantic caching, model routing and early stopping each cut LLM cost by 22–50% in single-agent studies. Nobody has measured what happens when you turn them on together inside a multi-agent pipeline, where one agent's output is the next agent's input. This repo is the harness for that measurement: a reference RAG + tool-use pipeline, the three techniques as switchable components, and a runner that compares configurations on cost, latency and task success. It also includes a learned alternative to the paper's hand-set router: a contextual bandit that learns the cost/quality trade-off from task outcomes.

## The pipeline

```mermaid
flowchart LR
    Q[Task] --> P[Orchestrator: plan]
    P --> R[Retrieval agent]
    R --> X[Reasoning agent + tools]
    X -- next subtask --> R
    X -- all done --> A[Orchestrator: aggregate]
    A --> F[Final answer]
```

Every LLM request from every node goes through one choke point (`Runtime.call_llm`) that routes it to a model, checks the cache, calls the provider and meters tokens and USD. Turning a technique on or off changes that choke point, not the agents.

| Config | Caching | Routing | Early stopping |
| --- | --- | --- | --- |
| `baseline` | | | |
| `cache_only` | ✓ | | |
| `route_only` | | ✓ | |
| `composed` | ✓ | ✓ | ✓ |

`configs/ablations/` adds configurations that isolate one mechanism each. None is in the paper's Table 1.

| Ablation | Extends | Change | Isolates |
| --- | --- | --- | --- |
| `cache_route` | `base` | caching + routing, no early stopping | H1 without a third technique |
| `early_stop_only` | `base` | early stopping only | early stopping's individual saving |
| `cache_only_pipeline` | `base` | `caching.scope: pipeline` | H2 (compare with `cache_only`) |
| `composed_model_aware` | `composed` | `caching.model_aware: true` | cross-model hits behind H1 |
| `composed_exclude_cached_drafts` | `composed` | `early_stopping.exclude_cached_drafts: true` | composition effect 4 |
| `composed_skip_tool_calls` | `composed` | `caching.skip_tool_calls: true` | savings from replayed action calls |
| `single_agent_baseline` | `base` | `mode: single_agent` | H3 single-agent arm, no techniques |
| `single_agent_route` | `single_agent_baseline` | routing on | H3 single-agent routing drop |
| `learned_route` | `route_only` | `routing.policy: linucb` | learned vs heuristic routing |
| `composed_learned` | `composed` | `routing.policy: linucb` | learned routing under composition |

`make run` runs what the paper needs: the four Table 1 configs plus `cache_route`, `cache_only_pipeline`, `composed_model_aware` and the two single-agent configs. `make run-extended` runs the other five ablations into the same output directory, and the report covers whatever has been run. Both targets use explicit lists in the Makefile, and a test fails if a file in `configs/ablations/` is in neither list or in both.

## What it measures

| Hypothesis | Mechanism instrumented | Where to look |
| --- | --- | --- |
| H1: caching + routing saves less than independent composition predicts | Cross-model cache hits: an answer cached from one model served to a request routed to another | `hypotheses.H1_sub_additive.primary_independence_test` (interaction term and 95% CI), `secondary_sum_of_savings`, `cross_model_hit_share` |
| H2: pipeline-level caching hits less than agent-level | Pipeline-scope keys include upstream agent outputs, so upstream drift changes the key | `hypotheses.H2_pipeline_vs_agent_cache`, `cache_hit_rate_by_agent` |
| H2, correctness side: hits that serve the wrong entry | Each cache entry stores the IDs, numbers and dates in its request; a hit whose entities differ from the incoming request is an entity-mismatch hit | `configs.<name>.cache_correctness` (hit rate, entity-mismatch share of hits, success given hit vs miss, per scope), "Cache correctness" in `summary.md` |
| H3: routing hurts high-complexity tasks more in multi-agent settings than in a single agent | Per-stratum success drop under routing for the multi-agent pipeline (`baseline` -> `route_only`) and a measured single agent (`single_agent_baseline` -> `single_agent_route`) | `hypotheses.H3_routing_success_drop` (`multi_agent`, `single_agent`, each with `high_minus_low`; `multi_minus_single`), all with 95% CIs |
| Effect 4 (not in the paper): cache-induced early stops | A cached draft replayed into a refinement step makes consecutive drafts agree and fires the stop rule. Counted when the stop fires on all drafts but would not fire with cached drafts removed | `cache_induced_early_stops` per task, `hypotheses.E4_cache_induced_early_stops`; ablation `composed_exclude_cached_drafts` |

Why H1 is tested against independence, not the sum. Let caching alone save a fraction a of baseline cost and routing alone save b. If the two act independently (caching removes a share of calls, routing makes the remaining calls cheaper), the composed saving is 1 − (1 − a)(1 − b) = a + b − ab. For any two positive savings that is below a + b, so composition is always sub-additive against the sum of savings, and finding it says nothing about interference. The primary H1 test is therefore against the independence prediction `1 − (1 − s_cache)(1 − s_route)`. The report gives the interaction term, observed composed saving minus the independence prediction, with a 95% paired bootstrap CI over tasks: a CI below zero means the techniques interfere, above zero means they reinforce each other, and a CI containing zero is consistent with independence. The comparison against the sum of savings, which is how the paper states H1, is kept as a secondary figure and labelled as such.

Per task the runner records the following. Everything after the first item is new in this version and defaults to 0 or null; existing configurations produce the same values for the original fields.

- Cost (at each model's price and at GPT-4o-equivalent price), wall-clock latency, success, LLM calls vs cache hits by agent, cross-model hits, routing decisions, reasoning iterations, early stops and tool calls.
- `entity_mismatch_hits`, `entity_mismatch_hits_by_agent`: cache hits served from an entry whose request had different IDs, numbers or dates.
- `cached_drafts`, `cached_draft_replays`, `cache_induced_early_stops`: reasoner drafts served from the cache, those repeating an earlier draft's answer in the same loop, and early stops that only fired because of cached drafts.
- `tool_call_responses_not_cached`: responses withheld from the cache by `caching.skip_tool_calls`.
- `mode`, `cache_scope`: pipeline mode and cache scope of the configuration.

Statistics. The headline comparisons are paired over tasks, because every configuration runs the same tasks with the same seeds. Success is compared with an exact McNemar test on (task, seed) pairs, plus the success-rate difference with a 95% paired bootstrap CI over tasks. Cost and latency differences use a paired bootstrap over tasks, with each task's value averaged over its seeds (10,000 resamples, fixed seed, 95% percentile CIs). Every config is compared with `baseline`, and each ablation with the config it modifies. The H3 strata are resampled within stratum. Seed-level mean ± std, P50/P95 latency and success by stratum are still reported as per-config descriptives. Every rate in the report is printed with its sample size.

## Learned routing (RL)

The paper's router scores each subtask with fixed weights and thresholds. `routing.policy: linucb` replaces it with a disjoint LinUCB contextual bandit: one arm per model on the ladder, context = the same four complexity features plus a bias term. It learns online, task by task, from

```
reward = 1[task succeeded] − λ · (call cost / same tokens at GPT-4o price)
```

with λ = `routing.bandit.cost_weight` (0.5 by default). Task success is only observed at the end, so every routing decision in a task shares it; the cost term is per call, so cost credit lands on the model that incurred it. Decisions answered from the cache are excluded from updates because the chosen model never ran.

Two questions this answers that the heuristic can't:

- Does a learned policy find a better cost/success frontier than hand-set thresholds? (`learned_vs_heuristic_routing` in the report)
- Does a policy trained under caching and early stopping learn around the interference behind H1? (`composed_learned` vs `composed`)

The report includes a learning curve (success and cost per quarter of the task stream) for each learned configuration, and the runner saves each policy's final state (per-arm pulls, mean reward, linear weights) to `results/<run>/policies/`. The weights show which complexity features the policy learned to route on.

## Quickstart (no API keys)

```bash
git clone https://github.com/ginoitaliano/cost-compose
cd cost-compose
pip install -e ".[dev]"
make test
make dry-run
```

The repository is named `cost-compose`. The Python distribution, the import path (`import costcompose`) and the CLI entry point (`costcompose`) are all `costcompose`, without the hyphen.

`--dry-run` swaps in a deterministic offline model and embedder. It exercises every code path (planning, tool calls, caching, routing, early stopping, reporting) in seconds. Its numbers are a plumbing check, not results.

## Full run

The experiments run on a Lightning AI Studio GPU, which also serves Llama-3-8B locally.

```bash
pip install -e ".[all]"
cp .env.example .env                       # OpenAI, Anthropic, LangSmith, Hugging Face keys
set -a; . ./.env; set +a                   # the CLI does not read .env itself; export it
docker run -d -p 6379:6379 redis:7-alpine  # cache backend
./scripts/serve_llama.sh &                 # vLLM, OpenAI-compatible, port 8000

# Task suite: 60 GAIA tasks + your enterprise RAG tasks, 40 per stratum in total
python scripts/prepare_gaia.py --per-stratum 20 --out data/tasks/gaia_subset.jsonl
cat data/tasks/gaia_subset.jsonl data/tasks/enterprise.jsonl > data/tasks/suite.jsonl

make run            # Table 1 + the ablations H1-H3 need, seeds 0 1 2 -> results/full/
make run-extended   # optional: the other ablations, appended to results/full/
make report         # rebuild summary.md / summary.json from runs.jsonl
make spotcheck      # 10% human spot-check sample -> results/full/spotcheck.csv
```

GAIA tasks come from the validation split only, because it is the only split with public answers. `prepare_gaia.py` excludes tasks the pipeline cannot attempt. Its tools are a calculator and search over the local corpus, so it drops tasks that need web browsing, tasks with an attached file it cannot read (any attachment by default; `--readable-extensions` allows some), and tasks that need multimodal input. The decision uses GAIA's metadata: the `file_name` field and the annotators' tool list, plus URLs or YouTube links in the question. Exclusion counts by reason are printed and written to `<out>.exclusions.json`, which holds counts only, no task content. Tasks with no annotator tool list are kept and counted so they can be reviewed by hand. If fewer than 20 eligible tasks remain in a stratum, the script warns rather than padding.

Human spot-checks (paper §3.4) check the automated scorer. `costcompose spotcheck sample results/full --tasks data/tasks/suite.jsonl` draws a seeded 10% of task IDs per complexity stratum (4 per stratum at 40/40/40) and one seed per sampled task. It writes a CSV row for every config that ran that task and seed, so scorer errors that depend on the config show up. It will not overwrite an existing CSV without `--force`. Fill `human_success` with 1 or 0, then `costcompose spotcheck score results/full/spotcheck.csv` reports agreement and Cohen's kappa against `auto_success`, overall and per config with n, and lists the disagreements. The CSV holds task questions, and for GAIA tasks it must not be published; `results/` is git-ignored.

Runs are resumable: `runs.jsonl` is appended one task at a time, and a rerun skips every (config, seed, task) already recorded. A failed task is logged with its error and scored as a failure rather than stopping the run.

Run any subset directly:

```bash
costcompose run --configs baseline composed --tasks data/tasks/suite.jsonl \
  --corpus data/corpus --seeds 0 1 2 --out results/full
costcompose validate --tasks data/tasks/suite.jsonl
```

## Configuration

All settings live in `configs/base.yaml`; each configuration `extends:` it and overrides only what differs. The paper's values:

| Setting | Value |
| --- | --- |
| Cache similarity threshold | 0.92 cosine, `text-embedding-3-small` |
| Cache scope | `agent` (per-agent namespace, local input) or `pipeline` (shared, input + upstream context) |
| Routing features | query length, tool-call count, context size, dependency depth; equal weights |
| Routing ladder | Llama-3-8B < 0.25 ≤ Haiku 4.5 < 0.5 ≤ GPT-4o < 0.75 ≤ Sonnet 4.5 |
| Low tier | score < 0.5 (Llama-3-8B, Haiku 4.5) |
| Early-stopping threshold | 0.88 on 0.5 × self-reported confidence + 0.5 × draft consistency (stop when it exceeds the threshold) |
| Router policy | `heuristic` (paper) or `linucb`; bandit α = 1.0, ridge = 1.0, λ = 0.5 |
| Seeds | 0, 1, 2 (seed also shuffles task order, which sets what the cache has seen) |

Settings added in this version. Each defaults to the paper setup, so existing configurations reproduce exactly.

| Setting | Default | Effect when changed |
| --- | --- | --- |
| `mode` | `multi_agent` | `single_agent`: one reasoning agent with `search_docs` and the calculator as functions, no orchestrator or retriever. Same `Runtime.call_llm` choke point and reasoning loop. Its dependency-depth routing feature is always 0. |
| `caching.skip_tool_calls` | `false` | `true`: a response containing a tool or function call with arguments is never stored or served. Plans, retrieval summaries and final text are still cached. |
| `caching.version_key` | `null` | Added to the cache namespace in both scopes, so a change invalidates earlier entries. `prompts` hashes the system prompts, `file:<path>` hashes a file, any other value is used as given. |
| `early_stopping.exclude_cached_drafts` | `false` | `true`: drafts served from the cache are left out of the stop rule (both the self-reported confidence and the consistency term). |

Prices are in `configs/base.yaml` as USD per million tokens. Check them against current provider pricing before a paid run, and set the self-hosted Llama price from your GPU rate and measured throughput.

Tasks are JSONL:

```json
{"id": "ent-042", "question": "...", "answer": "750", "complexity": "medium", "source": "enterprise"}
```

Scoring is GAIA-style quasi-exact match: numbers compared numerically, strings normalised, lists element-wise. Documents for retrieval go in `--corpus` as `.md` or `.txt`.

## Layout

```
src/costcompose/
  agents.py       runtime choke point + plan / retrieve / reason / aggregate / single-agent nodes
  pipeline.py     LangGraph and native engines (multi- and single-agent), single-task execution
  cache.py        semantic cache, agent/pipeline scope, entity extractor, memory/Redis backends
  routing.py      complexity features, heuristic router, LinUCB bandit router
  early_stop.py   confidence parsing, draft consistency, stop rule
  llm.py          LiteLLM clients + offline fake client and hashing embedder
  tracking.py     per-call cost and token accounting
  analysis.py     paired tests (McNemar, bootstrap), H1-H3 and effect 4, Markdown report
  experiment.py   resumable configs x seeds x tasks runner
  spotcheck.py    human spot-check sampling and scorer agreement
  cli.py          costcompose run | report | validate | spotcheck
configs/          Table 1 configurations and ablations
data/             sample corpus and tasks (GAIA is fetched, never committed)
scripts/          vLLM server, GAIA subset builder and filter
paper/            the paper and a section-to-code map
```

## Implementation notes

The cache is GPTCache-style, reimplemented for per-scope keys and cross-model hit accounting. It does not use the GPTCache library. It follows GPTCache's design (embed, nearest neighbour, cosine threshold, Redis store), but the study needs control GPTCache does not expose: per-agent vs pipeline-wide key construction, counting hits that cross model boundaries, entity-mismatch accounting and the tool-call guard.

Early stopping is a loop-level stop rule inspired by CALM (Schuster et al. 2022), not CALM itself. CALM is token-level early exit: for each generated token it decides how many decoder layers to run. Here the rule decides whether to run another iteration of an agent's reasoning loop, from the model's self-reported confidence and the agreement between consecutive drafts.

The entity extractor is a small set of regular expressions (dates, identifiers such as `ORD-48213` or `#5521`, numbers). It reads the agent's whole local input, including retrieved passages and tool transcripts, so it errs toward flagging a hit as a mismatch. It also misses differences its patterns do not cover, such as names. Self-reported `CONFIDENCE:` values are ignored. Read the entity-mismatch share as an indicator of wrong-entry hits, not as an exact count or a bound.

Both engines call the same node functions. LangGraph is the default when installed and is what LangSmith traces; the native engine exists so the core runs and tests with only NumPy and PyYAML. CI checks that both produce the same answers, call counts, cache hits, routing, early stops, cache-induced stops and entity-mismatch hits, in both `multi_agent` and `single_agent` mode.

Routing features use a 4-characters-per-token estimate. Billing uses the token counts the provider returns.

Semantic caching inside the reasoning loop can replay an earlier draft to a refinement step. By default that is counted, not suppressed: it is one of the composition effects the study is looking for (`cache_induced_early_stops`). `early_stopping.exclude_cached_drafts: true` suppresses it for the ablation.

The learned router pays for exploration early in the task stream: it has to try expensive models to learn when they are worth it. On a 120-task suite that is a real share of the run, so read its learning curve alongside the averages. Keep α at 1.0 or above; with rewards on this scale, smaller values can lock onto the first arm that looks acceptable.

## Results

Experiments are in progress. Results will be published here and in the paper.

## Citation

```bibtex
@inproceedings{altarouti2026composing,
  title     = {Composing Cost-Reduction Techniques in Multi-Agent {LLM} Systems: A Practitioner's Empirical Study},
  author    = {Altarouti, Sarah Ahmed},
  booktitle = {NeurIPS 2026 Workshop on Women in Machine Learning (WiML)},
  year      = {2026}
}
```

## License

MIT
