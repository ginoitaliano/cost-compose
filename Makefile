.PHONY: install dev test lint dry-run run run-extended report spotcheck docker clean

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
	costcompose run --dry-run --tasks data/tasks/sample.jsonl --corpus data/corpus --seeds 0 1 2 --out results/dry-run

# Configs the paper's Table 1 and hypotheses H1-H3 need.
CORE_CONFIGS = baseline cache_only route_only composed \
	configs/ablations/cache_route.yaml configs/ablations/cache_only_pipeline.yaml \
	configs/ablations/composed_model_aware.yaml \
	configs/ablations/single_agent_baseline.yaml configs/ablations/single_agent_route.yaml

# Ablations beyond the paper: individual early stopping, learned routing, composition effect 4,
# tool-call guard. Every file in configs/ablations/ must be in exactly one of the two lists
# (tests/test_config.py checks this).
EXTENDED_CONFIGS = configs/ablations/early_stop_only.yaml \
	configs/ablations/learned_route.yaml configs/ablations/composed_learned.yaml \
	configs/ablations/composed_exclude_cached_drafts.yaml \
	configs/ablations/composed_skip_tool_calls.yaml

RUN_ARGS = --tasks data/tasks/suite.jsonl --corpus data/corpus --seeds 0 1 2 --out results/full

# Full study, 3 seeds. Needs keys, Redis and the vLLM server.
run:
	costcompose -v run --configs $(CORE_CONFIGS) $(RUN_ARGS)

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
