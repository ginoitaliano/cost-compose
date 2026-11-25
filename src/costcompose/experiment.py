"""Run configurations x seeds x tasks and write results to disk (resumable)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from . import analysis
from .agents import Runtime
from .cache import InMemoryBackend, RedisBackend, SemanticCache, resolve_version_key
from .config import ExperimentConfig, load_config
from .early_stop import EarlyStopper
from .llm import FakeLLMClient, HashingEmbedder, LiteLLMClient, LiteLLMEmbedder
from .pipeline import build_langgraph, resolve_engine, run_task
from .retrieval import VectorStore
from .routing import make_router
from .tasks import Task, load_tasks, ordered_for_seed

log = logging.getLogger("costcompose")


def make_clients(backend: str, cfg: ExperimentConfig):
    if backend == "fake":
        return FakeLLMClient(), HashingEmbedder()
    if backend == "litellm":
        return LiteLLMClient(), LiteLLMEmbedder(cfg.embedding_model)
    raise ValueError(f"unknown backend {backend!r}")


def make_cache(
    cfg: ExperimentConfig, embedder, namespace: str, backend_override: str | None = None
) -> SemanticCache | None:
    c = cfg.caching
    if not c.enabled:
        return None
    if (backend_override or c.backend) == "redis":
        url = os.environ.get("COSTCOMPOSE_REDIS_URL", c.redis_url)
        backend = RedisBackend(url, prefix=f"costcompose:{namespace}")
    else:
        backend = InMemoryBackend()
    cache = SemanticCache(backend, embedder, c.threshold, c.scope, c.model_aware,
                          skip_tool_calls=c.skip_tool_calls,
                          version_key=resolve_version_key(c.version_key),
                          role_scoped=c.role_scoped, entity_guard=c.entity_guard)
    cache.reset()  # every (config, seed) run starts cold
    return cache


def _done_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            keys.add((r["config"], r["seed"], r["task_id"]))
    return keys


def _set_aside_errors(path: Path) -> int:
    """Move rows that ended in an error to runs.errors.jsonl so a rerun retries them.

    A task that raised (an unreachable endpoint, a rate limit that outlasted the retries, a
    provider rejecting a request) says nothing about the configuration, and keeping it would
    count an outage as a task failure. The set-aside rows are appended to runs.errors.jsonl, so
    nothing is lost; a task that errors again on the rerun is recorded again as a failure.
    """
    if not path.exists():
        return 0
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    failed = [r for r in rows if r.get("error")]
    if failed:
        with (path.parent / "runs.errors.jsonl").open("a") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in failed)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows if not r.get("error")))
    return len(failed)


class PreflightError(RuntimeError):
    pass


def models_used(cfg: ExperimentConfig) -> list[str]:
    """Every generation model a configuration can call."""
    if cfg.routing.enabled:
        return [r.model for r in cfg.routing.ladder]
    return [cfg.routing.default_model]


def preflight(configs: list[ExperimentConfig], llm, embedders: dict[str, Any]) -> list[str]:
    """Call each model and embedder the run will use once, before task 1.

    Raises PreflightError naming every model that fails, so a dead endpoint or missing key stops
    the run in seconds instead of turning into a column of task failures.
    """
    models = sorted({m for c in configs for m in models_used(c)})
    problems = []
    for m in models:
        try:
            llm.complete(m, [{"role": "user", "content": "Reply with OK."}], temperature=0.0)
        except Exception as exc:  # report every failure at once
            hint = ""
            if m.startswith("hosted_vllm/") and not os.environ.get("HOSTED_VLLM_API_BASE"):
                hint = " (HOSTED_VLLM_API_BASE is not set)"
            problems.append(f"{m}: {type(exc).__name__}: {str(exc)[:200]}{hint}")
    for name, emb in embedders.items():
        try:
            emb.embed(["ok"])
        except Exception as exc:
            problems.append(f"{name} (embeddings): {type(exc).__name__}: {str(exc)[:200]}")
    if problems:
        raise PreflightError("preflight failed:\n  " + "\n  ".join(problems))
    return models


def run_experiment(
    config_paths: list[str | Path],
    tasks: list[Task],
    corpus_dir: str | Path | None,
    seeds: list[int],
    out_dir: str | Path,
    backend: str = "litellm",
    engine: str = "auto",
    cache_backend: str | None = None,
    retry_errors: bool = True,
    check_models: bool = True,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    if retry_errors:
        moved = _set_aside_errors(runs_path)
        if moved:
            log.warning("retrying %d task runs that ended in an error (kept in %s)",
                        moved, runs_path.parent / "runs.errors.jsonl")
    done = _done_keys(runs_path)
    engine = resolve_engine(engine)
    configs = [load_config(p) for p in config_paths]
    if check_models and backend != "fake":
        llm0, _ = make_clients(backend, configs[0])
        embedders = {c.embedding_model: make_clients(backend, c)[1] for c in configs}
        checked = preflight(configs, llm0, embedders)
        log.info("preflight ok: %s", ", ".join(checked))

    stores: dict[str, VectorStore] = {}
    with runs_path.open("a") as fh:
        for cfg in configs:
            llm, embedder = make_clients(backend, cfg)
            key = f"{backend}:{cfg.embedding_model}"
            if key not in stores:  # index once; indexing cost is not attributed to tasks
                stores[key] = VectorStore.from_directory(corpus_dir, embedder) if corpus_dir \
                    else VectorStore(embedder)
            for seed in seeds:
                rt = Runtime(
                    cfg=cfg,
                    llm=llm,
                    router=make_router(cfg.routing),
                    stopper=EarlyStopper(cfg.early_stopping),
                    store=stores[key],
                    cache=make_cache(cfg, embedder, f"{cfg.name}:{seed}", cache_backend),
                    seed=seed,
                )
                graph = build_langgraph(rt) if engine == "langgraph" else None
                ordered = ordered_for_seed(tasks, seed)
                todo = [(i, t) for i, t in enumerate(ordered)
                        if (cfg.name, seed, t.id) not in done]
                log.info("%s seed=%d: %d tasks (%s engine)", cfg.name, seed, len(todo), engine)
                if todo and len(todo) < len(ordered) and rt.router.policy != "heuristic":
                    log.warning("%s seed=%d: resuming a learned router restarts it from "
                                "scratch; delete its rows to rerun the seed cleanly",
                                cfg.name, seed)
                for position, task in todo:
                    rec = run_task(rt, task, engine=engine, graph=graph)
                    rec.update(config=cfg.name, seed=seed, backend=backend, position=position,
                               techniques=cfg.techniques, mode=cfg.mode,
                               cache_scope=cfg.caching.scope if cfg.caching.enabled else None)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    if rec["error"]:
                        log.warning("%s/%s failed: %s", cfg.name, task.id, rec["error"])
                if todo and rt.router.policy != "heuristic":
                    pdir = out / "policies"
                    pdir.mkdir(exist_ok=True)
                    state = json.dumps(rt.router.state_dict(), indent=2)
                    (pdir / f"{cfg.name}_seed{seed}.json").write_text(state)
    return write_report(out)


def write_report(out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    records = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines() if x.strip()]
    summary = analysis.summarize(records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(analysis.to_markdown(summary))
    return summary


__all__ = ["run_experiment", "write_report", "load_tasks"]
