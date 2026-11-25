"""MuSiQue suite builder, tested on synthetic records in MuSiQue's format (no download)."""

import importlib.util
import json

from helpers import ROOT

from costcompose.retrieval import VectorStore
from costcompose.llm import HashingEmbedder
from costcompose.tasks import load_tasks

_spec = importlib.util.spec_from_file_location("prepare_musique",
                                               ROOT / "scripts" / "prepare_musique.py")
mq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mq)


def _rec(i, hops, answerable=True, shared_distractor=False):
    paras = [{"idx": j, "title": f"Title {i}-{j}", "paragraph_text": f"Text {i}-{j}.",
              "is_supporting": j < hops} for j in range(20)]
    if shared_distractor:
        paras[-1] = {"idx": 19, "title": "Shared", "paragraph_text": "Same everywhere.",
                     "is_supporting": False}
    return {"id": f"{hops}hop__{i}", "question": f"Question {i}?", "answer": f"Answer {i}",
            "answer_aliases": [f"Alias {i}", f"Answer {i}"], "answerable": answerable,
            "paragraphs": paras,
            "question_decomposition": [{"id": k, "question": "q", "answer": "a",
                                        "paragraph_support_idx": k} for k in range(hops)]}


RECORDS = ([_rec(i, 2) for i in range(10)] + [_rec(100 + i, 3) for i in range(10)]
           + [_rec(200 + i, 4) for i in range(10)]
           + [_rec(300, 5), _rec(301, 2, answerable=False), {"id": "broken"}])


def test_classify_and_exclusions():
    _, log = mq.select(RECORDS, {"low": 1, "medium": 1, "high": 1}, seed=0)
    assert log["rows"] == 33
    assert log["excluded"] == {"unsupported_hops": 1, "unanswerable": 1, "missing_fields": 1}
    assert log["eligible_by_stratum"] == {"low": 10, "medium": 10, "high": 10}


def test_fills_only_what_other_sources_lack_and_is_prefix_stable():
    big, _ = mq.select(RECORDS, {"low": 6, "medium": 4, "high": 2}, seed=0)
    small, log = mq.select(RECORDS, {"low": 3, "medium": 0, "high": 2}, seed=0)
    assert log["written_by_stratum"] == {"low": 3, "medium": 0, "high": 2}
    for s in ("low", "high"):
        assert [r["id"] for r in small[s]] == [r["id"] for r in big[s]][: len(small[s])]
    other, _ = mq.select(RECORDS, {"low": 6, "medium": 4, "high": 2}, seed=1)
    assert [r["id"] for r in other["low"]] != [r["id"] for r in big["low"]]


def test_task_format_and_aliases():
    t = mq.to_task(_rec(7, 3), "medium")
    assert t["id"] == "musique-3hop__7" and t["source"] == "musique-3hop"
    assert t["aliases"] == ["Alias 7"]  # the answer itself is not repeated


def test_end_to_end_files_load_as_tasks_and_corpus(tmp_path):
    src = tmp_path / "dev.jsonl"
    recs = [_rec(i, 2, shared_distractor=True) for i in range(4)] + [_rec(100, 3)]
    src.write_text("".join(json.dumps(r) + "\n" for r in recs))
    have = tmp_path / "gaia.jsonl"
    have.write_text(json.dumps({"id": "g1", "question": "q", "answer": "1",
                                "complexity": "low"}) + "\n")
    out, corpus = tmp_path / "mq.jsonl", tmp_path / "corpus" / "musique"
    (corpus).mkdir(parents=True)
    (corpus / "stale.md").write_text("old")
    mq.main(["--input", str(src), "--have", str(have), "--per-stratum", "3",
             "--out", str(out), "--corpus-dir", str(corpus)])

    tasks = load_tasks(out)
    assert [t.complexity for t in tasks] == ["low", "low", "medium"]  # 3 - 1 GAIA low
    assert tasks[0].aliases and tasks[0].answers[0] == tasks[0].answer
    log = json.loads((tmp_path / "mq.jsonl.log.json").read_text())
    assert log["have_by_stratum"] == {"low": 1, "medium": 0, "high": 0}
    assert log["needed_by_stratum"]["high"] == 3 and log["written_by_stratum"]["high"] == 0
    assert len(log["input_sha256"]) == 64
    # 3 tasks x 20 paragraphs, the shared distractor stored once
    assert log["paragraphs"] == 60 and log["unique_paragraphs_written"] == 59
    assert not (corpus / "stale.md").exists()
    store = VectorStore.from_directory(corpus.parent, HashingEmbedder())
    assert len(store.passages) == 59


def _hop_rec(i, hops, ids):
    r = _rec(i, hops)
    r["question_decomposition"] = [{"id": h, "question": "q", "answer": "a",
                                    "paragraph_support_idx": k} for k, h in enumerate(ids)]
    return r


# 12 two-hop and 12 three-hop questions; hops 1000-1003 are each shared by three questions,
# everything else is unique.
REUSE = ([_hop_rec(i, 2, [1000 + i % 4, 5000 + i]) for i in range(12)]
         + [_hop_rec(100 + i, 3, [1000 + i % 4, 6000 + i, 7000 + i]) for i in range(12)])


def test_reuse_share_zero_is_the_plain_selection():
    need = {"low": 5, "medium": 5, "high": 0}
    plain, _ = mq.select(REUSE, need, seed=0)
    same, _ = mq.select_reuse(REUSE, need, seed=0, share=0.0)
    for s in ("low", "medium"):
        assert [r["id"] for r in same[s]] == [r["id"] for r in plain[s]]


def test_reuse_share_draws_clusters_that_share_a_hop():
    need = {"low": 6, "medium": 6, "high": 0}
    picked, log = mq.select_reuse(REUSE, need, seed=0, share=0.5, cluster_max=3)
    assert log["written_by_stratum"] == {"low": 6, "medium": 6, "high": 0}
    assert log["clustered_by_stratum"] == {"low": 3, "medium": 3, "high": 0}
    assert all(2 <= len(c["tasks"]) <= 3 for c in log["clusters"])
    selected = [r for s in ("low", "medium") for r in picked[s]]
    assert len({r["id"] for r in selected}) == 12  # no task picked twice
    for c in log["clusters"]:
        members = [r for r in selected if f"musique-{r['id']}" in c["tasks"]]
        assert all(c["hop"] in mq.hop_ids(r) for r in members)
    assert mq.hop_sharing(selected)["tasks_sharing_a_hop"] >= 6
    again, _ = mq.select_reuse(REUSE, need, seed=0, share=0.5, cluster_max=3)
    assert [r["id"] for r in again["low"]] == [r["id"] for r in picked["low"]]


def test_reuse_stream_end_to_end_logs_sharing(tmp_path):
    src = tmp_path / "dev.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in REUSE))
    out, corpus = tmp_path / "reuse.jsonl", tmp_path / "corpus_reuse" / "musique"
    mq.main(["--input", str(src), "--per-stratum", "4", "--reuse-share", "1.0",
             "--out", str(out), "--corpus-dir", str(corpus)])
    assert len(load_tasks(out)) == 8
    log = json.loads((tmp_path / "reuse.jsonl.log.json").read_text())
    assert log["reuse_share"] == 1.0 and log["clusters"]
    assert log["hop_sharing"]["tasks_sharing_a_hop"] == 8
