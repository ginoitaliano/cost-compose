.PHONY: install dev test lint dry-run suite suite-reuse suite-dev run-dev run-dev-refs run run-extended run-reuse report spotcheck docker clean

# LiteLLM logs every call at INFO; keep errors only (override: make run LITELLM_LOG=INFO).
export LITELLM_LOG ?= ERROR

install:
	pip install -e ".[all]"

dev:
	pip install -e ".[dev,all]"
	pre-commit install || true

test:
	pytest --cov=costcompose

lint:
	ruff check .

dry-run:
	costcompose run --dry-run --tasks data/tasks/sample.jsonl --corpus data/corpus/sample --seeds 0 1 2 --out results/dry-run

# Task suite: every eligible GAIA task, the enterprise tasks if written, and MuSiQue filling each
# stratum to 40. Needs HF_TOKEN with GAIA access; MuSiQue is fetched from a Hugging Face mirror
# unless MUSIQUE_INPUT points at the official musique_ans_v1.0_dev.jsonl.
ENTERPRISE = $(wildcard data/tasks/enterprise.jsonl)
MUSIQUE_INPUT ?=
suite:
	python scripts/prepare_gaia.py --per-stratum 40 --out data/tasks/gaia_subset.jsonl
	python scripts/prepare_musique.py $(if $(MUSIQUE_INPUT),--input $(MUSIQUE_INPUT)) \
		--have data/tasks/gaia_subset.jsonl $(ENTERPRISE) --per-stratum 40 \
		--out data/tasks/musique_subset.jsonl --corpus-dir data/corpus/musique
	cat data/tasks/gaia_subset.jsonl $(ENTERPRISE) data/tasks/musique_subset.jsonl \
		> data/tasks/suite.jsonl
	costcompose validate --tasks data/tasks/suite.jsonl

# Reuse stream: same GAIA (and enterprise) tasks, same strata and sizes, but half of each
# stratum's MuSiQue tasks come in clusters of 2-3 questions that share a single-hop question, so
# some subtasks recur across tasks. Separate corpus directory, so the main suite is unchanged.
REUSE_SHARE ?= 0.5
suite-reuse:
	python scripts/prepare_musique.py $(if $(MUSIQUE_INPUT),--input $(MUSIQUE_INPUT)) \
		--have data/tasks/gaia_subset.jsonl $(ENTERPRISE) --per-stratum 40 \
		--reuse-share $(REUSE_SHARE) --out data/tasks/musique_reuse.jsonl \
		--corpus-dir data/corpus_reuse/musique
	cat data/tasks/gaia_subset.jsonl $(ENTERPRISE) data/tasks/musique_reuse.jsonl \
		> data/tasks/suite_reuse.jsonl
	costcompose validate --tasks data/tasks/suite_reuse.jsonl

# Configs the paper's Table 1 and hypotheses H1-H3 need, plus the guarded-cache arm.
CORE_CONFIGS = baseline cache_only route_only composed \
	configs/ablations/cache_route.yaml configs/ablations/cache_only_pipeline.yaml \
	configs/ablations/composed_model_aware.yaml \
	configs/ablations/single_agent_baseline.yaml configs/ablations/single_agent_route.yaml \
	configs/ablations/cache_only_guarded.yaml configs/ablations/composed_guarded.yaml \
	configs/ablations/cache_route_guarded.yaml \
	configs/ablations/cache_only_structured.yaml configs/ablations/composed_structured.yaml \
	configs/ablations/cache_route_structured.yaml \
	configs/ablations/llama_only.yaml configs/ablations/haiku_only.yaml \
	configs/ablations/sonnet_only.yaml

# Ablations beyond the paper: individual early stopping, learned routing, composition effect 4,
# tool-call guard, one cache guard at a time. Every file in configs/ablations/ must be in
# exactly one of the two lists (tests/test_config.py checks this).
EXTENDED_CONFIGS = configs/ablations/early_stop_only.yaml \
	configs/ablations/learned_route.yaml configs/ablations/composed_learned.yaml \
	configs/ablations/composed_exclude_cached_drafts.yaml \
	configs/ablations/composed_skip_tool_calls.yaml \
	configs/ablations/cache_only_role_scoped.yaml configs/ablations/cache_only_first_step.yaml \
	configs/ablations/cache_only_entity_guard.yaml \
	configs/ablations/cache_only_skip_tool_calls.yaml \
	configs/ablations/cache_only_structured_keys.yaml

RUN_ARGS = --tasks data/tasks/suite.jsonl --corpus data/corpus --seeds 0 1 2 --out results/full

# Full study, 3 seeds. Needs keys, Redis and the vLLM server.
run:
	costcompose -v run --configs $(CORE_CONFIGS) $(RUN_ARGS)

# Development set: MuSiQue only, disjoint from both evaluation suites, 20 per stratum, with the
# reuse stream's clustering so cache methods have reuse to work with. All method changes are made
# on this set; the evaluation suites are run once. Build suite and suite-reuse first.
suite-dev:
	python scripts/prepare_musique.py $(if $(MUSIQUE_INPUT),--input $(MUSIQUE_INPUT)) \
		--per-stratum 20 --reuse-share $(REUSE_SHARE) --seed 1 \
		--exclude data/tasks/musique_subset.jsonl data/tasks/musique_reuse.jsonl \
		--out data/tasks/dev.jsonl --corpus-dir data/corpus_dev/musique
	costcompose validate --tasks data/tasks/dev.jsonl

# Development runs: the base pipeline and the caches, one seed. Read retrieval with
# scripts/retrieval_check.py pipeline --tasks data/tasks/dev.jsonl --runs results/dev/runs.jsonl
DEV_CONFIGS = baseline cache_only configs/ablations/cache_only_guarded.yaml \
	configs/ablations/cache_only_structured.yaml configs/ablations/cache_only_structured_keys.yaml
run-dev:
	costcompose -v run --configs $(DEV_CONFIGS) --tasks data/tasks/dev.jsonl \
		--corpus data/corpus_dev --seeds 0 --out results/dev

# Single models, the oracle bound and routing on the development set, into the same directory.
DEV_REF_CONFIGS = baseline route_only configs/ablations/llama_only.yaml \
	configs/ablations/haiku_only.yaml configs/ablations/sonnet_only.yaml
run-dev-refs:
	costcompose -v run --configs $(DEV_REF_CONFIGS) --tasks data/tasks/dev.jsonl \
		--corpus data/corpus_dev --seeds 0 --out results/dev

# Caching configs on the reuse stream, into their own output directory.
REUSE_CONFIGS = baseline cache_only route_only composed \
	configs/ablations/cache_route.yaml configs/ablations/cache_only_pipeline.yaml \
	configs/ablations/cache_only_guarded.yaml configs/ablations/composed_guarded.yaml \
	configs/ablations/cache_route_guarded.yaml configs/ablations/cache_only_structured.yaml \
	configs/ablations/composed_structured.yaml configs/ablations/cache_route_structured.yaml
run-reuse:
	costcompose -v run --configs $(REUSE_CONFIGS) --tasks data/tasks/suite_reuse.jsonl \
		--corpus data/corpus_reuse --seeds 0 1 2 --out results/full_reuse

# Extra ablations. Same output directory: runs are appended and the report covers both.
run-extended:
	costcompose -v run --configs $(EXTENDED_CONFIGS) $(RUN_ARGS)

report:
	costcompose report results/full

# Human spot-check sample (10% of tasks, stratified) from the full run; grade it, then
# `costcompose spotcheck score results/full/spotcheck.csv`.
spotcheck:
	costcompose spotcheck sample results/full --tasks data/tasks/suite.jsonl

docker:
	docker build -t costcompose .
	docker run --rm costcompose

clean:
	rm -rf .pytest_cache .ruff_cache .coverage build dist src/*.egg-info results/dry-run
