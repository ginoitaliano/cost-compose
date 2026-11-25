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

Correctness side of H2: every stored entry keeps the entities (IDs, numbers, dates) extracted
from the agent's local input. On a hit, the cache compares them with the entities in the
incoming request; a difference marks an ``entity_mismatch`` hit, the clearest case of a
semantically close but wrong answer (same question, different order ID).

``skip_tool_calls``: when True, a response that contains a tool/function call with arguments
is never stored and never served. Replaying an action call is riskier than replaying text: its
arguments were chosen for the request that produced it. Plans, retrieval summaries and final
text are still cached.

``version_key``: optional instruction/context version. When set, it becomes part of the
namespace in both scopes, so entries written under other instructions are never matched. It goes
in the namespace rather than the embedded text because a hash appended to the text moves cosine
similarity only slightly and would not guarantee a miss.

Guards for caching inside an agent pipeline. With all of them off the cache behaves like a
GPTCache-style cache wired to each agent's input, which is how semantic caches are usually
dropped into agent frameworks; that is the naive arm of the study. Each guard follows a
practice from deployed semantic caches, adapted to a multi-agent pipeline:

* ``role_scoped``: one namespace per role, not per agent. The orchestrator plans and
  aggregates; without this, an aggregation request can be served the stored plan.
* ``first_step_only`` (enforced by the runtime, which knows loop positions): only the first
  step of a reasoning loop is looked up or stored. Later steps differ from earlier ones by a
  tool result or a refinement request, a small change to a long input, so without this a step
  can be served the response its own loop produced one step earlier.
* ``entity_guard``: a hit is refused when the entities (IDs, numbers, dates) of the request's
  query text differ from those of the stored request. The query text is the question or
  subtask, not the retrieved passages, whose entities differ almost every time.
* ``skip_tool_calls`` (above).

Structured keys (``structured=True``). The text key embeds the agent's whole input, in which the
part that says what is asked (a subtask line) is short and the shared context (task, passages,
earlier results) is long, so different requests in one task look alike and the same request in
two tasks looks different. A structured key embeds only what is asked (the question or subtask)
and puts what the answer depends on (retrieved document IDs, upstream answers, loop history,
entities of the ask) in the namespace as an exact digest, with one namespace per role. A hit then
needs a request that means the same thing and rests on exactly the same inputs: semantic match on
the ask, exact match on its dependencies. The runtime supplies both parts; see ``dependency_key``.

Every entry records where it was written (task run, reasoning loop, role) so the runtime can
classify each hit: served back into the same loop, from elsewhere in the same task, or from an
earlier task; and whether it crossed roles or models.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .llm import Embedder, LLMResponse

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = re.compile(
    rf"\b(?:\d{{4}}-\d{{1,2}}-\d{{1,2}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}"
    rf"|{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:,?\s+\d{{4}})?)\b",
    re.IGNORECASE,
)
# Identifiers: letters followed by digits (ORD-48213, INV_77, A12B), or #12345.
_ID = re.compile(r"\b[A-Za-z]+[-_]?\d[A-Za-z0-9_-]*\b|(?<!\w)#\d+\b")
# Self-reported confidence in replayed drafts is reasoning-loop scaffolding, not an entity of
# the request; without this, every refinement step would look like a different query.
_SCAFFOLD = re.compile(r"CONFIDENCE:\s*[0-9.]+", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")


def extract_entities(text: str) -> frozenset[str]:
    """Regex entity extractor: dates, then identifiers, then bare numbers.

    Each match is removed before the next pattern runs so a date's digits are not also counted
    as numbers. Values are normalised (lower-case IDs, numbers without thousands separators and
    trailing zeros) so formatting differences do not count as different entities.
    """
    found: set[str] = set()
    rest = _SCAFFOLD.sub(" ", text)
    for kind, pattern in (("date", _DATE), ("id", _ID)):
        for m in pattern.finditer(rest):
            found.add(f"{kind}:{' '.join(m.group(0).lower().split())}")
        rest = pattern.sub(" ", rest)
    for m in _NUMBER.finditer(rest):
        value = float(m.group(0).replace(",", ""))
        found.add(f"num:{value:g}")
    return frozenset(found)


def dependency_key(*parts) -> str:
    """Short exact digest of what a response depends on (JSON-serialisable parts)."""
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _has_tool_call_args(tool_calls) -> bool:
    """True if any tool call carries arguments. Accepts ToolCall objects or their dicts."""
    for tc in tool_calls or []:
        args = tc.get("arguments") if isinstance(tc, dict) else tc.arguments
        if args:
            return True
    return False


def resolve_version_key(value: str | None) -> str | None:
    """Turn ``caching.version_key`` into the string used in cache namespaces.

    * ``None``: no versioning (default; namespaces unchanged).
    * ``"prompts"``: short hash of every system prompt in ``costcompose.prompts``.
    * ``"file:<path>"``: short hash of that file's contents (e.g. an instructions file).
    * any other string: used as given (e.g. a hash computed elsewhere, or "v2").
    """
    if value is None:
        return None
    value = str(value)  # YAML may give `version_key: 2` as an int
    if value == "prompts":
        from . import prompts

        text = "\n".join(
            f"{k}={v}" for k, v in sorted(vars(prompts).items())
            if k.isupper() and isinstance(v, str)
        )
        return "prompts-" + hashlib.sha256(text.encode()).hexdigest()[:12]
    if value.startswith("file:"):
        data = Path(value[len("file:"):]).read_bytes()
        return "file-" + hashlib.sha256(data).hexdigest()[:12]
    return value


@dataclass
class CacheHit:
    response: LLMResponse
    similarity: float
    source_model: str
    entity_mismatch: bool = False  # served entry came from a request with different entities
    query_entity_mismatch: bool = False  # ...judged on the query text only (the guard's test)
    origin: dict | None = None  # where the entry was written: {"run", "loop", "role"}


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
    entity_mismatch_hits: Counter = field(default_factory=Counter)
    tool_call_entries_refused: Counter = field(default_factory=Counter)  # hits not served
    entity_guard_refusals: Counter = field(default_factory=Counter)  # hits not served

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
        skip_tool_calls: bool = False,
        version_key: str | None = None,
        role_scoped: bool = False,
        entity_guard: bool = False,
        structured: bool = False,
    ) -> None:
        if scope not in {"agent", "pipeline"}:
            raise ValueError(scope)
        self.backend = backend
        self.embedder = embedder
        self.threshold = threshold
        self.scope = scope
        self.model_aware = model_aware
        self.skip_tool_calls = skip_tool_calls
        self.version_key = version_key
        self.role_scoped = role_scoped
        self.entity_guard = entity_guard
        self.structured = structured
        self.stats = CacheStats()

    def _namespace(
        self, agent: str, model: str, role: str | None = None, deps: str | None = None
    ) -> str:
        ns = f"agent:{agent}" if self.scope == "agent" else "pipeline"
        if self.role_scoped or self.structured:
            ns = f"{ns}|role:{role or agent}"
        if deps is not None:
            ns = f"{ns}|dep:{deps}"
        if self.version_key is not None:
            ns = f"{ns}|version:{self.version_key}"
        return f"{ns}|model:{model}" if self.model_aware else ns

    def key_text(self, agent: str, local_input: str, pipeline_context: str) -> str:
        if self.structured or self.scope == "agent":
            return local_input
        return f"[{agent}]\n{pipeline_context}\n---\n{local_input}"

    def lookup(
        self,
        agent: str,
        model: str,
        local_input: str,
        pipeline_context: str,
        role: str | None = None,
        entity_text: str | None = None,
        deps: str | None = None,
    ) -> tuple[CacheHit | None, np.ndarray, int]:
        """Return (hit or None, query embedding, embedding tokens billed).

        ``role`` names the agent's role for this request (planner, aggregator, ...), used for
        namespaces when ``role_scoped`` and for classifying hits. ``entity_text`` is the query
        text whose entities the guard compares (defaults to ``local_input``).
        """
        vecs, tokens = self.embedder.embed([self.key_text(agent, local_input, pipeline_context)])
        vec = vecs[0]
        self.stats.lookups[agent] += 1
        found = self.backend.nearest(self._namespace(agent, model, role, deps), vec)
        if found is None or found[0] < self.threshold:
            return None, vec, tokens
        sim, payload = found
        if self.skip_tool_calls and _has_tool_call_args(payload["response"].get("tool_calls")):
            self.stats.tool_call_entries_refused[agent] += 1
            return None, vec, tokens
        query = local_input if entity_text is None else entity_text
        stored_q = payload.get("query_entities")
        q_mismatch = stored_q is not None and frozenset(stored_q) != extract_entities(query)
        if self.entity_guard and q_mismatch:
            self.stats.entity_guard_refusals[agent] += 1
            return None, vec, tokens
        self.stats.hits[agent] += 1
        if payload["model"] != model:
            self.stats.cross_model_hits[agent] += 1
        stored = payload.get("entities")  # absent on entries written before this field existed
        mismatch = stored is not None and frozenset(stored) != extract_entities(local_input)
        if mismatch:
            self.stats.entity_mismatch_hits[agent] += 1
        hit = CacheHit(LLMResponse.from_dict(payload["response"]), sim, payload["model"],
                       mismatch, q_mismatch, payload.get("origin"))
        return hit, vec, tokens

    def store(
        self,
        agent: str,
        model: str,
        vector: np.ndarray,
        response: LLMResponse,
        local_input: str | None = None,
        role: str | None = None,
        entity_text: str | None = None,
        origin: dict | None = None,
        deps: str | None = None,
    ) -> bool:
        """Store a response. Returns False if it was withheld (``skip_tool_calls``)."""
        if self.skip_tool_calls and _has_tool_call_args(response.tool_calls):
            return False
        payload = {"model": model, "agent": agent, "response": response.to_dict()}
        if local_input is not None:
            payload["entities"] = sorted(extract_entities(local_input))
            query = local_input if entity_text is None else entity_text
            payload["query_entities"] = sorted(extract_entities(query))
        if origin is not None:
            payload["origin"] = origin
        self.backend.add(self._namespace(agent, model, role, deps), vector, payload)
        return True

    def reset(self) -> None:
        self.backend.clear()
        self.stats = CacheStats()
