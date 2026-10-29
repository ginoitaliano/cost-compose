.PHONY: install dev test lint dry-run run report docker clean

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

# Full study: Table 1 configs plus ablations, 3 seeds. Needs keys, Redis and the vLLM server.
run:
	costcompose -v run --configs baseline cache_only route_only composed configs/ablations/*.yaml \
		--tasks data/tasks/suite.jsonl --corpus data/corpus --seeds 0 1 2 --out results/full

report:
	costcompose report results/full

docker:
	docker build -t costcompose .
	docker run --rm costcompose

clean:
	rm -rf .pytest_cache .ruff_cache .coverage build dist src/*.egg-info results/dry-run
