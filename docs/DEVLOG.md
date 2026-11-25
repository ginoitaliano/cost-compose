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

## Next

- Measure the noise floor: baseline on the development set with a second seed.
- Rerun the development set after the prompt change; read the oracle bound.
- Write and commit the preregistration, then run the evaluation suites once.
