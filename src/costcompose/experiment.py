"""Run configurations x seeds x tasks and write results to disk (resumable)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from . import analysis
from .agents import Runtime
from .cache import InMemoryBackend, RedisBackend, SemanticCache
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
    cache = SemanticCache(backend, embedder, c.threshold, c.scope, c.model_aware)
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


def run_experiment(
    config_paths: list[str | Path],
    tasks: list[Task],
    corpus_dir: str | Path | None,
    seeds: list[int],
    out_dir: str | Path,
    backend: str = "litellm",
    engine: str = "auto",
    cache_backend: str | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    done = _done_keys(runs_path)
    engine = resolve_engine(engine)
    configs = [load_config(p) for p in config_paths]

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
                               techniques=cfg.techniques)
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
