# Contributing

```bash
make dev        # editable install with all extras + pre-commit
make test       # offline: no API keys needed
make lint
```

Everything must pass offline. Code that calls a provider goes behind the `LLMClient` / `Embedder` protocols in `src/costcompose/llm.py` so tests can swap in the fake backend.

New pipeline configurations go in `configs/` (Table 1) or `configs/ablations/` and should `extends:` an existing file rather than copy it. Add a test in `tests/test_config.py` if a configuration encodes a value the paper states.

Open an issue before large changes to the pipeline graph: node behaviour feeds directly into the reported metrics.
