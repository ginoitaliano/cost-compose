# Development log

Every change to the method or the pipeline made after data was first seen, what prompted it, and
what it changed. The evaluation suites (`data/tasks/suite.jsonl`, `data/tasks/suite_reuse.jsonl`)
are run once, after the preregistration is committed; everything below was decided on pilots or on
the development set (`data/tasks/dev.jsonl`). This log is reproduced in the paper's appendix.

Dates are UTC. Numbers are from single-seed runs unless stated, so differences of a few
percentage points are within run-to-run noise.

## 2026-11-24. Smoke tests: answer scoring

- Seen: GPT-4o appended units to 8 of 9 correct answers ("95 USD" for a question "in USD").
- Changed: a primary score that accepts unit words the question names, next to the strict
  GAIA-style score; the aggregator prompt asks for the bare number. Both scores reported.

## 2026-11-24. Suite: GAIA shortfall

- Seen: 18 of 165 GAIA validation tasks are usable without browsing, attachments or multimodal
  input (8 low, 6 medium, 4 high).
- Changed: MuSiQue-Ans dev questions fill each stratum to 40, stratified by hop count.

## 2026-11-25. Pilot 1 (12 tasks from the evaluation suite): cache false hits

These pilot tasks came from the evaluation suite. Of its 18 GAIA tasks, 8 were seen in pilots.

- Seen: the agent-scope cache saved 57.8% of cost while success fell from 33.3% to 16.7%.
  Success given a hit was 1.8% (n=56). Predictions included the first line of a JSON plan
  (the aggregator was served the planner's stored output) and one task made 100 tool calls
  (reasoning-loop steps were served their own previous tool call).
- Seen: no request scored at or above the top routing cutoff (0.75); routing raised cost 13.4%.
- Changed: hit-origin accounting (same loop, same task, cross task, cross role); four opt-in
  cache guards (role-scoped namespaces, first loop step only, entity guard, no tool-call
  caching) as a second arm; routing scores recorded per request. Defaults unchanged.

## 2026-11-25. Pilot 2 (same tasks): guarded cache

- Seen: with the guards on, the hit rate fell from 40.9% to 3.7%, no hit came from an earlier
  task, and the cost saving fell from 57.8% to 5.9% (CI includes 0). The naive saving was
  within-task false hits; unrelated questions give a correct cache nothing to reuse.
- Changed: a reuse stream in which half of each stratum's MuSiQue tasks come in clusters that
  share a single-hop question (`--reuse-share 0.5`).

## 2026-11-25. Reuse pilot (14 tasks)

- Seen: the guarded cache found reuse across tasks (3 of 7 hits, 2 of the 3 in successful
  tasks) but too little to change cost. Its 4 within-task hits were one subtask's answer served
  to another subtask of the same task; all 4 tasks failed.
- Seen: pipeline scope hit more often than agent scope (44.6% vs 34.3%), against H2's
  prediction, through cross-role hits in its shared namespace; success given a hit was 3.2%.
- Decided: method work moves to a development set disjoint from both evaluation suites.

## 2026-11-25. Development set and retrieval

- Changed: `make suite-dev` (20 MuSiQue questions per stratum, reuse clustering, own corpus).
- Found and fixed: documents whose first paragraph exceeded about 790 characters were chunked
  into a title-only passage and an untitled one, which affected most MuSiQue paragraphs. Every
  run before this fix had degraded retrieval.
- Changed: run records list retrieved document IDs; `scripts/retrieval_check.py`.

## 2026-11-25. Development run 1 (60 tasks): caches and retrieval

- Seen: support recall 78%; baseline success 30% when every supporting paragraph was retrieved
  and 43% when some were missed. Retrieval is not the bottleneck.
- Seen: of the 21 baseline failures with full retrieval, 8 were answers deserving partial
  credit ("1999" for "30 November 1999"), 5 were abstentions ("Unknown", "Not available") and
  8 were wrong.
- Seen, structured cache: all 21 hits came from earlier tasks (no within-task false hits), but
  every task with a hit failed. By role: reasoner 15, retriever 4, planner 1, aggregator 1.
  Clustered questions share paragraphs, so different subtasks over the same documents pass the
  exact dependency match, and short subtask texts clear 0.92 similarity.
- Pre-declared stopping rule: one further cache change only if the planner caused the hits.
  It did not, so cache design is frozen: naive, guarded and structured are all reported.
- Changed: answer F1 (MuSiQue's official metric) on every record; prompts ask for a best answer
  instead of "unknown" and for the answer rather than a sentence (changes every configuration's
  requests; runs before and after are not comparable); single-model reference configs and the
  oracle bound.

## 2026-11-25. Development run 2 (after the prompt change): serving limit

- Seen: 3 of 60 `llama_only` task runs failed with ContextWindowExceededError. The vLLM server
  was started with `--max-model-len 8192`, a serving choice, while Llama 3.1 supports 131,072
  tokens and every other model on the ladder accepts 128k or more.
- Changed: `serve_llama.sh` serves Llama 3.1 at its native 131,072 tokens and uses vLLM's
  default generation config (`--generation-config vllm`) rather than the model repository's
  sampling defaults, so the request's temperature and seed apply as for the API models. The
  failed runs are retried on resume.

- Seen after the change: one `llama_only` task timed out at 120 s per attempt. With no output
  cap, vLLM allows a generation up to the remaining context (now about 130k tokens), and a
  degenerate repeating output runs until the timeout. The 8,192 limit had masked this.
- Changed: `agents.max_output_tokens: 4096`, sent with every request to every model. Answers,
  drafts and plans are far shorter, so the cap only stops runaway generations.

## 2026-11-25. Development run 3 (two seeds for the reference configurations)

- Seen: run-to-run spread is larger than one baseline suggested. Success varies by 2.4 pp
  between seeds for `baseline` and 8.2 pp for `route_only` (17.7 pp for `haiku_only` in the low
  stratum). Differences under about 8 pp on the development set are not distinguishable from
  rerunning.
- Seen: per task, Claude Haiku 4.5 costs 59% more than GPT-4o ($0.034 vs $0.022) at a third of
  its list price per token, through more calls (13.8 vs 10.1) and more tokens. The heuristic
  router sends 54% of requests to Haiku and raises cost by 48%.
- Seen: the per-run oracle saving (49.4%) fell to 36.7% [13.5, 55.8] when averaged over two
  seeds. Its picks: no model solves the task 37%, GPT-4o 27%, Haiku 18%, Sonnet 10%, Llama 8%.
- Seen: an iso-success oracle (never worse than GPT-4o on any task, tasks GPT-4o never solves
  left on GPT-4o) saves 3.4% [1.0, 6.3]. Sending the tasks GPT-4o never solves to Llama instead
  saves 67.2% [54.7, 78.5]. Routing headroom at equal quality is small; the large saving is in
  tasks no model solves.
- Decided: no cascade router. Its ceiling at equal quality is about 3%. Predicting task failure
  is named in the paper as the open problem.
- Decided: the evaluation run is fixed in `PREREGISTRATION.md`. The analysis adds the
  iso-success oracle, tokens per task, bootstrap p-values and a Holm-corrected confirmatory set
  (C1-C4). The composed guarded and structured configurations and `composed_model_aware` move to
  the extended set; the main run has 15 configurations and the reuse run 9.

- Decided: Llama is priced at a serverless list price ($0.05 / $0.08 per million tokens, Groq,
  checked 2026-11-25), like every other model, instead of from our GPU's hourly rate. Measured
  vLLM throughput while serving the development runs one request at a time was 258 tokens per
  second (187 prompt, 71 generation), which puts the self-hosted price near GPT-4o's input
  price; providers batch many requests and charge 30-100 times less. The self-hosted price is
  reported as a sensitivity check.

## Next

- Commit the preregistration, then run both evaluation streams once.
