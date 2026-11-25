# Preregistration: evaluation run

This file fixes the evaluation run before it happens. It is committed, and its commit hash is
quoted in the paper, before any task in the evaluation suites is run for the final results.
Anything done differently afterwards is a deviation: it is logged in `docs/DEVLOG.md` with its
date and reason, and the paper reports it.

Everything decided before this point was decided on pilots or on the development set
(`data/tasks/dev.jsonl`); `docs/DEVLOG.md` lists each decision and the data behind it.

## Question

In a multi-agent LLM pipeline, how much do semantic caching and model routing save, alone and
composed, once savings are measured against (a) what the cache actually served, (b) what the
cheapest correct model would have cost, and (c) what independent techniques should save
together?

## Data

| Stream | File | Tasks | Built by |
| --- | --- | --- | --- |
| Main | `data/tasks/suite.jsonl` | 120: 18 GAIA validation (8 low, 6 medium, 4 high) and 102 MuSiQue-Ans dev, 40 per stratum | `make suite` |
| Reuse | `data/tasks/suite_reuse.jsonl` | 120: the same 18 GAIA tasks and 102 MuSiQue-Ans dev, half of each stratum's MuSiQue tasks in clusters of 2-3 that share a single-hop question | `make suite-reuse` |

The MuSiQue selections are seeded (seed 0) and logged with the input file's SHA-256 next to each
task file. The development set is disjoint from both.

Tasks seen before this preregistration, all disclosed in the paper: the first pilot used 12
tasks of the main suite (2 GAIA and 2 MuSiQue per stratum) and the reuse pilot 14 tasks of the
reuse suite (6 GAIA and 8 MuSiQue). These tasks stay in the suites. The paper also reports the
confirmatory estimates with them excluded:
`costcompose report results/full --exclude-tasks results/pilot/tasks.jsonl results/pilot/reuse_tasks.jsonl`
writes `summary.excluded.md`.

## Fixed settings

Everything in `configs/base.yaml` at the commit that adds this file, including:

- Models: Llama-3.1-8B-Instruct (vLLM, 131,072-token context, `--generation-config vllm`),
  Claude Haiku 4.5, GPT-4o, Claude Sonnet 4.5; embeddings `text-embedding-3-small`.
- Paper values: cosine threshold 0.92, routing ladder cutoffs 0.25 / 0.5 / 0.75, early-stop
  threshold 0.88, seeds 0, 1, 2, strata 40/40/40.
- Temperature 0.2; output cap 4,096 tokens per request; 5 reasoning iterations per subtask;
  4 subtasks; top-4 retrieval.
- API list prices, checked 2026-11-25: GPT-4o $2.50 / $10.00, Claude Haiku 4.5 $1 / $5,
  Claude Sonnet 4.5 $3 / $15 per million input / output tokens; `text-embedding-3-small`
  $0.02 per million tokens.
- The Llama price per million tokens is set before the run from the Studio GPU's hourly rate
  and the throughput vLLM reports during the development runs:
  price = hourly rate / (3,600 x tokens per second) x 10^6, the same for input and output.
  The value and its inputs are recorded in `configs/base.yaml` and `docs/DEVLOG.md`.

## Configurations

Main stream, seeds 0, 1, 2 (`make run`, or `make run-a`, `run-b`, `run-c` and `make merge`):

`baseline`, `cache_only`, `route_only`, `composed`, `cache_route`, `cache_only_pipeline`,
`single_agent_baseline`, `single_agent_route`, `cache_only_guarded`, `cache_route_guarded`,
`cache_only_structured`, `cache_route_structured`, `llama_only`, `haiku_only`, `sonnet_only`.

Reuse stream, seeds 0, 1, 2 (`make run-reuse`):

`baseline`, `cache_only`, `route_only`, `cache_route`, `cache_only_pipeline`,
`cache_only_guarded`, `cache_route_guarded`, `cache_only_structured`, `cache_route_structured`.

Configurations in `EXTENDED_CONFIGS` are not part of this run. If run later they are reported as
exploratory and separately.

## Outcomes

- Primary: task success under the primary score (quasi-exact match that accepts unit words the
  question names, against the answer or any listed alias); cost per task in USD at list prices.
- Secondary: strict success (GAIA-style quasi-exact match); answer F1 (MuSiQue's official
  metric); latency; LLM calls, input tokens and output tokens per task.

## Confirmatory tests

Four tests on the main stream, all two-sided, Holm-corrected together at alpha = 0.05. The report
prints them first (`hypotheses.confirmatory`).

| ID | Estimand | Test | Direction seen on the development set |
| --- | --- | --- | --- |
| C1 | Success, `cache_only` minus `baseline` | exact McNemar on (task, seed) pairs | lower (-11.7 pp) |
| C2 | Cost, `cache_only` relative to `baseline` | paired bootstrap over tasks | lower (-37%) |
| C3 | H1 interaction: observed `cache_route` saving minus the independence prediction 1 - (1 - s_cache)(1 - s_route) | paired bootstrap over tasks | not run on the development set |
| C4 | Cost, `route_only` relative to `baseline` | paired bootstrap over tasks | higher (+48%) |

A confirmatory claim is made only if its Holm-adjusted p-value is below 0.05. The paper states
each result in the direction observed, whichever it is.

## Estimands reported without a hypothesis test

Pre-declared and reported with 95% CIs and sample sizes, for both streams where they apply. They
are described, not tested.

1. Cache mechanisms: hit rate, share of hits by origin (same loop, same task, cross task),
   cross-role and entity-mismatch shares, success given a hit by origin, for the naive, guarded
   and structured caches; the subtask cache's hit rate.
2. Guarded and structured caches against `baseline` and against `cache_only`: success and cost
   differences.
3. H1 for the guarded and structured caches (`H1_guarded_cache`, `H1_structured_cache`).
4. H2: agent-scope against pipeline-scope hit rate and entity-mismatch share.
5. H3: per-stratum routing success drop for the multi-agent pipeline and the single agent, and the
   difference-of-differences.
6. Oracle bounds over the single-model configurations: per-run, seed-averaged, iso-success (never
   worse than `baseline` on any task), and iso-success with tasks `baseline` never solves sent to
   the cheapest model. Routed configurations as a share of the seed-averaged bound.
7. Cost per task by model, with calls and tokens per task, for the single-model configurations.
8. Cache-induced early stops in `composed`.

## Analysis

The analysis is the code in `src/costcompose/analysis.py` at this commit, run with
`costcompose report`. Paired comparisons over tasks; success with an exact McNemar test on
(task, seed) pairs; cost, latency, F1 and the H1 interaction with a paired bootstrap over tasks,
each task averaged over its seeds, 10,000 resamples, resampling seed 0, 95% percentile
intervals. If a bug is found in the analysis after the run, the fixed analysis is reported next
to the original and the fix is logged as a deviation.

## Errors and exclusions

A task run that raises an error (provider outage, rate limit after retries) is retried once by
rerunning the same command, which moves errored runs to `runs.errors.jsonl` and runs them again.
A run that errors again is scored as a failure and counted in the report. No task is excluded
after the run.

## Run once

Each stream is run once with the configurations above. No configuration is added, removed or
changed after results are seen. The development set remains available for exploratory work,
reported as such.

## Budget

From the development runs, with 30% added for the GAIA tasks, which cost more than MuSiQue tasks:
about $170 for the main stream, about 22 hours of wall clock in one process or about 9 hours as
three parallel parts, and about $100 for the reuse stream.
