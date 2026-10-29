"""LLM and embedding clients.

Two backends share one interface:

* ``LiteLLMClient`` / ``LiteLLMEmbedder`` call real providers through LiteLLM.
* ``FakeLLMClient`` / ``HashingEmbedder`` are deterministic and offline. They exist so the
  whole pipeline, the tests and CI run without API keys. Numbers they produce are plumbing
  checks, never results.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LLMResponse:
        return cls(
            content=d["content"],
            model=d["model"],
            usage=Usage(**d.get("usage", {})),
            tool_calls=[ToolCall(**tc) for tc in d.get("tool_calls", [])],
        )


class LLMClient(Protocol):
    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> LLMResponse: ...


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        """Return (L2-normalised embeddings [n, d], tokens billed)."""
        ...


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token). Used for routing features only."""
    return max(1, len(text) // 4)


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# --------------------------------------------------------------------------- real backends


class LiteLLMClient:
    """Thin wrapper over ``litellm.completion`` with retries and tool-call parsing."""

    def __init__(self, max_retries: int = 3, timeout_s: float = 120.0) -> None:
        import litellm  # imported lazily so the package works without the extra

        self._litellm = litellm
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    def complete(self, model, messages, tools=None, temperature=0.0, seed=None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.timeout_s,
            "num_retries": self.max_retries,
            "drop_params": True,  # providers that ignore `seed` etc. won't error
        }
        if tools:
            kwargs["tools"] = tools
        if seed is not None:
            kwargs["seed"] = seed
        resp = self._litellm.completion(**kwargs)
        msg = resp.choices[0].message
        calls = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            content=msg.content or "",
            model=model,
            usage=Usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            tool_calls=calls,
        )


class LiteLLMEmbedder:
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        import litellm

        self._litellm = litellm
        self.model = model

    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        resp = self._litellm.embedding(model=self.model, input=texts)
        vecs = np.array([d["embedding"] for d in resp.data], dtype=np.float32)
        tokens = getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0
        return _normalise(vecs), tokens


# --------------------------------------------------------------------------- offline backends

_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (no network, no model weights)."""

    model = "hashing-256"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _WORD.findall(text.lower()):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
                out[i, h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        return _normalise(out), sum(approx_tokens(t) for t in texts)


class FakeLLMClient:
    """Deterministic stand-in for a provider.

    It reads the role tag the pipeline puts in each system prompt (``[role=...]``) and emits
    output in the format that role expects, so every code path (planning, tool calls,
    confidence parsing, early stopping, aggregation) is exercised offline.
    """

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, model, messages, tools=None, temperature=0.0, seed=None) -> LLMResponse:
        self.calls += 1
        system = messages[0]["content"] if messages else ""
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        role = re.search(r"\[role=(\w+)\]", system)
        role = role.group(1) if role else "generic"
        digest = int(hashlib.sha256(f"{model}|{user}|{seed}".encode()).hexdigest(), 16)
        prompt_tokens = sum(approx_tokens(str(m.get("content", ""))) for m in messages)

        tool_calls: list[ToolCall] = []
        if role == "planner":
            n = 1 + min(2, approx_tokens(user) // 40)
            content = json.dumps([f"Step {i + 1}: {user[:80]}" for i in range(n)])
        elif role == "retriever":
            content = "RELEVANT: " + user[:200]
        elif role == "reasoner":
            already_used_tool = any(m["role"] == "tool" for m in messages)
            expr = re.search(r"(\d+(?:\.\d+)?\s*[-+*/]\s*\d+(?:\.\d+)?)", user)
            iteration = sum(1 for m in messages if m["role"] == "assistant")
            if tools and expr and not already_used_tool:
                tool_calls = [
                    ToolCall(id=f"call_{digest % 10_000}", name="calculator",
                             arguments={"expression": expr.group(1)})
                ]
                content = ""
            else:
                conf = min(0.99, 0.6 + 0.15 * iteration + (digest % 10) / 100)
                status = "FINAL" if iteration >= 3 else "DRAFT"
                content = f"ANSWER: draft for {user[:40]}\nCONFIDENCE: {conf:.2f}\nSTATUS: {status}"
        elif role == "aggregator":
            content = "FINAL ANSWER: unknown"
        else:
            content = "OK"

        completion_tokens = approx_tokens(content) + 8 * len(tool_calls)
        return LLMResponse(
            content=content,
            model=model,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            tool_calls=tool_calls,
        )
