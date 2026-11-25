"""Minimal in-memory vector store over a directory of text/markdown documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .llm import Embedder


@dataclass
class Passage:
    doc_id: str
    text: str


def chunk(text: str, max_chars: int = 800) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        # A heading alone is never a chunk: it stays with the text that follows it. Without
        # this, a document whose first paragraph is long splits into a title-only passage and
        # an untitled one (most Wikipedia paragraphs in the MuSiQue corpus).
        heading_only = buf.startswith("#") and "\n" not in buf
        if buf and not heading_only and len(buf) + len(p) + 2 > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks


class VectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.passages: list[Passage] = []
        self._matrix: np.ndarray | None = None
        self.index_tokens = 0

    def add_documents(self, docs: dict[str, str]) -> None:
        new = [Passage(doc_id, c) for doc_id, text in docs.items() for c in chunk(text)]
        if not new:
            return
        vecs, tokens = self.embedder.embed([p.text for p in new])
        self.index_tokens += tokens
        self.passages.extend(new)
        self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])

    @classmethod
    def from_directory(cls, path: str | Path, embedder: Embedder) -> VectorStore:
        store = cls(embedder)
        files = sorted(Path(path).glob("**/*")) if path else []
        docs = {f.stem: f.read_text() for f in files if f.suffix in {".md", ".txt"}}
        store.add_documents(docs)
        return store

    def search(self, query: str, k: int = 4) -> tuple[list[Passage], int]:
        """Return (top-k passages, embedding tokens billed for the query)."""
        if self._matrix is None:
            return [], 0
        vecs, tokens = self.embedder.embed([query])
        sims = self._matrix @ vecs[0]
        idx = np.argsort(-sims)[:k]
        return [self.passages[i] for i in idx], tokens
