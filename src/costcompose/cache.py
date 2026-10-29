"""Semantic cache with explicit scope control.

Lookup follows GPTCache's design: embed the request, find the nearest stored request by cosine
similarity, and return its response if similarity >= threshold (0.92 in the paper). The cache is
implemented here, rather than wrapped, because the study needs control that GPTCache does not
expose directly:

* ``scope="agent"``: one namespace per agent, keyed on the agent's *local* input only.
* ``scope="pipeline"``: one shared namespace, keyed on the agent's input *plus* the context
  accumulated upstream in the pipeline. Upstream drift changes the key, which is the mechanism
  behind hypothesis H2.
* ``model_aware``: when False, an entry produced by one model can answer a request routed to
  another. Those cross-model hits are counted separately because they are the interference
  mechanism behind hypothesis H1.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .llm import Embedder, LLMResponse


@dataclass
class CacheHit:
    response: LLMResponse
    similarity: float
    source_model: str


class VectorBackend(Protocol):
    def add(self, namespace: str, vector: np.ndarray, payload: dict) -> None: ...

    def nearest(self, namespace: str, vector: np.ndarray) -> tuple[float, dict] | None: ...

    def clear(self) -> None: ...


class InMemoryBackend:
    def __init__(self) -> None:
        self._vecs: dict[str, list[np.ndarray]] = {}
        self._payloads: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def add(self, namespace, vector, payload):
        with self._lock:
            self._vecs.setdefault(namespace, []).append(vector.astype(np.float32))
            self._payloads.setdefault(namespace, []).append(payload)

    def nearest(self, namespace, vector):
        with self._lock:
            vecs = self._vecs.get(namespace)
            if not vecs:
                return None
            sims = np.stack(vecs) @ vector
            i = int(np.argmax(sims))
            return float(sims[i]), self._payloads[namespace][i]

    def clear(self):
        with self._lock:
            self._vecs.clear()
            self._payloads.clear()


class RedisBackend:
    """Redis-backed store. Vectors live in Redis; similarity search is brute force in NumPy,
    which is exact and fast enough for experiment-scale caches (thousands of entries)."""

    def __init__(self, url: str, prefix: str = "costcompose:cache") -> None:
        import redis  # lazy: only needed when backend == "redis"

        self._r = redis.Redis.from_url(url)
        self._prefix = prefix

    def _key(self, namespace: str) -> str:
        return f"{self._prefix}:{namespace}"

    def add(self, namespace, vector, payload):
        record = {"v": vector.astype(np.float32).tolist(), "p": payload}
        self._r.rpush(self._key(namespace), json.dumps(record))

    def nearest(self, namespace, vector):
        raw = self._r.lrange(self._key(namespace), 0, -1)
        if not raw:
            return None
        records = [json.loads(x) for x in raw]
        sims = np.array([r["v"] for r in records], dtype=np.float32) @ vector
        i = int(np.argmax(sims))
        return float(sims[i]), records[i]["p"]

    def clear(self):
        for key in self._r.scan_iter(f"{self._prefix}:*"):
            self._r.delete(key)


@dataclass
class CacheStats:
    lookups: Counter = field(default_factory=Counter)
    hits: Counter = field(default_factory=Counter)
    cross_model_hits: Counter = field(default_factory=Counter)

    def hit_rate(self, agent: str | None = None) -> float:
        lk = self.lookups[agent] if agent else sum(self.lookups.values())
        ht = self.hits[agent] if agent else sum(self.hits.values())
        return ht / lk if lk else 0.0


class SemanticCache:
    def __init__(
        self,
        backend: VectorBackend,
        embedder: Embedder,
        threshold: float = 0.92,
        scope: str = "agent",
        model_aware: bool = False,
    ) -> None:
        if scope not in {"agent", "pipeline"}:
            raise ValueError(scope)
        self.backend = backend
        self.embedder = embedder
        self.threshold = threshold
        self.scope = scope
        self.model_aware = model_aware
        self.stats = CacheStats()

    def _namespace(self, agent: str, model: str) -> str:
        ns = f"agent:{agent}" if self.scope == "agent" else "pipeline"
        return f"{ns}|model:{model}" if self.model_aware else ns

    def key_text(self, agent: str, local_input: str, pipeline_context: str) -> str:
        if self.scope == "agent":
            return local_input
        return f"[{agent}]\n{pipeline_context}\n---\n{local_input}"

    def lookup(
        self, agent: str, model: str, local_input: str, pipeline_context: str
    ) -> tuple[CacheHit | None, np.ndarray, int]:
        """Return (hit or None, query embedding, embedding tokens billed)."""
        vecs, tokens = self.embedder.embed([self.key_text(agent, local_input, pipeline_context)])
        vec = vecs[0]
        self.stats.lookups[agent] += 1
        found = self.backend.nearest(self._namespace(agent, model), vec)
        if found is None or found[0] < self.threshold:
            return None, vec, tokens
        sim, payload = found
        self.stats.hits[agent] += 1
        if payload["model"] != model:
            self.stats.cross_model_hits[agent] += 1
        hit = CacheHit(LLMResponse.from_dict(payload["response"]), sim, payload["model"])
        return hit, vec, tokens

    def store(self, agent: str, model: str, vector: np.ndarray, response: LLMResponse) -> None:
        payload = {"model": model, "agent": agent, "response": response.to_dict()}
        self.backend.add(self._namespace(agent, model), vector, payload)

    def reset(self) -> None:
        self.backend.clear()
        self.stats = CacheStats()
