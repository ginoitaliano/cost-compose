"""Retrieval diagnostics for MuSiQue tasks: is a failure a retrieval miss or a reasoning error?

Two checks, both using the <tasks>.support.json file that prepare_musique.py writes next to a
task file (supporting-paragraph document IDs and resolved single-hop sub-questions per task).

Ceiling (no LLM calls): embed the corpus and search it with (a) the full question and (b) each
gold single-hop sub-question with earlier answers filled in, the query a perfect planner would
issue. Reports recall@k of the supporting paragraphs per stratum. Costs only embeddings.

    python scripts/retrieval_check.py ceiling --tasks data/tasks/dev.jsonl \
        --corpus data/corpus_dev [--dry-run]

Pipeline (no LLM calls): read a run's runs.jsonl, where every record lists the document IDs each
search returned, and report per config the share of supporting paragraphs the pipeline ever
retrieved, and success when all were retrieved vs when some were missed.

    python scripts/retrieval_check.py pipeline --tasks data/tasks/dev.jsonl \
        --runs results/dev/runs.jsonl

Both print Markdown and write a JSON file next to their input (--out to change it).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from costcompose.llm import HashingEmbedder, LiteLLMEmbedder
from costcompose.retrieval import VectorStore
from costcompose.tasks import load_tasks

ORDER = ("low", "medium", "high")


def _support(tasks_path: Path) -> dict:
    path = Path(f"{tasks_path}.support.json")
    if not path.exists():
        sys.exit(f"{path} not found; rebuild the task file with scripts/prepare_musique.py")
    return json.loads(path.read_text())


def _rate(k: float, n: float) -> dict:
    return {"rate": float(k / n) if n else None, "n": int(n)}


def _fmt(x: dict) -> str:
    return "–" if x["rate"] is None else f"{100 * x['rate']:.1f}% (n={x['n']})"


def ranked_doc_ids(store: VectorStore, query: str, depth: int) -> list[str]:
    """Distinct document IDs in rank order (a document can have several chunks)."""
    passages, _ = store.search(query, k=depth * 4)
    out: list[str] = []
    for p in passages:
        if p.doc_id not in out:
            out.append(p.doc_id)
        if len(out) == depth:
            break
    return out


def ceiling(tasks_path: Path, corpus: Path, ks: list[int], embedder) -> dict:
    sup = _support(tasks_path)
    tasks = [t for t in load_tasks(tasks_path) if t.id in sup]
    store = VectorStore.from_directory(corpus, embedder)
    depth = max(ks)
    by = {q: {s: defaultdict(lambda: [0, 0]) for s in ORDER} for q in ("question", "subquestion")}
    for t in tasks:
        docs = sup[t.id]["support_docs"]
        ranked = ranked_doc_ids(store, t.question, depth)
        for k in ks:  # question as query: share of all supporting paragraphs in the top k
            c = by["question"][t.complexity][k]
            c[0] += sum(d in ranked[:k] for d in docs)
            c[1] += len(docs)
        # each hop's sub-question against the paragraph that answers that hop
        for sq, d in zip(sup[t.id]["subquestions"], sup[t.id]["hop_support_docs"], strict=True):
            if d is None:
                continue
            ranked = ranked_doc_ids(store, sq, depth)
            for k in ks:
                c = by["subquestion"][t.complexity][k]
                c[0] += d in ranked[:k]
                c[1] += 1
    out = {q: {s: {str(k): _rate(*by[q][s][k]) for k in ks} for s in ORDER} for q in by}
    out["corpus_documents"] = len({p.doc_id for p in store.passages})
    out["corpus_chunks"] = len(store.passages)
    out["tasks"] = len(tasks)
    return out


def ceiling_markdown(res: dict, ks: list[int]) -> str:
    lines = [f"# Retrieval ceiling ({res['tasks']} tasks, {res['corpus_documents']} documents, "
             f"{res['corpus_chunks']} chunks)", "",
             "Recall@k of supporting paragraphs. Question: the full multi-hop question as the "
             "query. Sub-question: each gold single-hop question with earlier answers filled "
             "in, the query a perfect planner would issue. n is supporting paragraphs.", "",
             "| Query | Stratum | " + " | ".join(f"@{k}" for k in ks) + " |",
             "| --- | --- | " + " | ".join("---" for _ in ks) + " |"]
    for q in ("question", "subquestion"):
        for s in ORDER:
            cells = [_fmt(res[q][s][str(k)]) for k in ks]
            lines.append(f"| {q} | {s} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pipeline(tasks_path: Path, runs_path: Path) -> dict:
    sup = _support(tasks_path)
    rows = [json.loads(x) for x in runs_path.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r["task_id"] in sup and "retrieved_doc_ids" in r]
    acc: dict = defaultdict(lambda: {"found": 0, "docs": 0, "all": [0, 0], "missed": [0, 0],
                                     "by": {s: [0, 0] for s in ORDER}})
    for r in rows:
        docs = set(sup[r["task_id"]]["support_docs"])
        got = {d for ids in r["retrieved_doc_ids"] for d in ids}
        a = acc[r["config"]]
        hit = len(docs & got)
        a["found"] += hit
        a["docs"] += len(docs)
        a["by"][r["complexity"]][0] += hit
        a["by"][r["complexity"]][1] += len(docs)
        bucket = a["all"] if hit == len(docs) else a["missed"]
        bucket[0] += bool(r["success"])
        bucket[1] += 1
    return {
        c: {
            "support_recall": _rate(a["found"], a["docs"]),
            "support_recall_by_stratum": {s: _rate(*a["by"][s]) for s in ORDER},
            "task_runs_all_support_retrieved": a["all"][1],
            "success_all_retrieved": _rate(*a["all"]),
            "success_some_missed": _rate(*a["missed"]),
        }
        for c, a in sorted(acc.items())
    }


def pipeline_markdown(res: dict) -> str:
    lines = ["# Retrieval in the pipeline", "",
             "Support recall: share of supporting paragraphs returned by any search in the task "
             "run (n = supporting paragraphs). Success is split by whether every supporting "
             "paragraph was retrieved (n = task runs).", "",
             "| Config | Support recall | low | medium | high | Success, all retrieved "
             "| Success, some missed |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for c, r in res.items():
        by = r["support_recall_by_stratum"]
        lines.append(f"| {c} | {_fmt(r['support_recall'])} | {_fmt(by['low'])} "
                     f"| {_fmt(by['medium'])} | {_fmt(by['high'])} "
                     f"| {_fmt(r['success_all_retrieved'])} | {_fmt(r['success_some_missed'])} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("ceiling")
    c.add_argument("--tasks", type=Path, required=True)
    c.add_argument("--corpus", type=Path, required=True)
    c.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    c.add_argument("--embedding-model", default="text-embedding-3-small")
    c.add_argument("--dry-run", action="store_true", help="offline hashing embedder")
    c.add_argument("--out", type=Path)
    p = sub.add_parser("pipeline")
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "ceiling":
        emb = HashingEmbedder() if args.dry_run else LiteLLMEmbedder(args.embedding_model)
        ks = sorted(set(args.k))
        res = ceiling(args.tasks, args.corpus, ks, emb)
        md = ceiling_markdown(res, ks)
        out = args.out or Path(f"{args.tasks}.retrieval_ceiling.json")
    else:
        res = pipeline(args.tasks, args.runs)
        md = pipeline_markdown(res)
        out = args.out or args.runs.parent / "retrieval_pipeline.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(md)
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
