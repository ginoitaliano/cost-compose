FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[all]"

COPY configs ./configs
COPY data ./data
RUN useradd --create-home runner && mkdir -p results && chown runner results
USER runner

ENTRYPOINT ["costcompose"]
CMD ["run", "--dry-run", "--tasks", "data/tasks/sample.jsonl", "--corpus", "data/corpus", "--seeds", "0", "--out", "results"]
