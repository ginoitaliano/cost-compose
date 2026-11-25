"""Build the MuSiQue portion of the task suite and its retrieval corpus.

MuSiQue (Trivedi et al., TACL 2022; CC BY 4.0) is multi-hop question answering over Wikipedia
paragraphs. Every question is composed from 2, 3 or 4 single-hop questions, each answered by one
paragraph, so later hops depend on earlier answers. Hop count gives the complexity strata:

    2 hops -> low, 3 hops -> medium, 4 hops -> high

Only the dev split of MuSiQue-Ans is used: the test split's answers are not public, and the Ans
variant contains answerable questions only. The script fills each stratum up to --per-stratum
after counting the tasks already in the --have files (the GAIA subset, the enterprise tasks), so
MuSiQue makes up whatever the other sources lack. Selection is a seeded shuffle per stratum: a
smaller selection is always a prefix of a larger one, so adding enterprise tasks later only drops
MuSiQue tasks from the end of each stratum's list.

Each selected question's 20 paragraphs (supporting plus distractors) are written, deduplicated,
to --corpus-dir as one Markdown file per paragraph. All questions share one corpus, so retrieval
also has to rule out other questions' paragraphs.

    # official release (recommended): bash download_data.sh in StonyBrookNLP/musique, then
    python scripts/prepare_musique.py --input path/to/musique_ans_v1.0_dev.jsonl \
        --have data/tasks/gaia_subset.jsonl --per-stratum 40
    # or let it fetch the dev file from a Hugging Face mirror (pip install huggingface_hub)
    python scripts/prepare_musique.py --have data/tasks/gaia_subset.jsonl --per-stratum 40

The SHA-256 of the input file is logged so the exact data can be cited and checked. Outputs are
git-ignored like the GAIA subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

STRATA = {2: "low", 3: "medium", 4: "high"}
ORDER = ("low", "medium", "high")
REQUIRED = ("id", "question", "answer", "paragraphs", "question_decomposition")
MIRROR_REPO = "dgslibisey/MuSiQue"
MIRROR_FILE = "musique_ans_v1.0_dev.jsonl"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def count_have(paths: list[Path]) -> Counter:
    have: Counter = Counter()
    for p in paths:
        for row in load(p):
            have[row["complexity"]] += 1
    return have


def classify(rec: dict) -> tuple[str | None, str | None]:
    """(stratum, None) for a usable record, or (None, exclusion reason)."""
    if any(k not in rec for k in REQUIRED):
        return None, "missing_fields"
    if rec.get("answerable") is False:
        return None, "unanswerable"
    hops = len(rec["question_decomposition"])
    if hops not in STRATA:
        return None, "unsupported_hops"
    return STRATA[hops], None


def select(records: list[dict], need: dict[str, int], seed: int) -> tuple[dict, dict]:
    """Seeded selection per stratum. Returns (selected by stratum, log)."""
    pools: dict[str, list[dict]] = {s: [] for s in ORDER}
    excluded: Counter = Counter()
    for rec in records:
        stratum, reason = classify(rec)
        if reason:
            excluded[reason] += 1
        else:
            pools[stratum].append(rec)
    picked = {}
    for s in ORDER:
        pool = sorted(pools[s], key=lambda r: r["id"])
        random.Random(f"{seed}:{s}").shuffle(pool)
        picked[s] = pool[: max(0, need.get(s, 0))]
    log = {
        "rows": len(records),
        "excluded": dict(excluded),
        "eligible_by_stratum": {s: len(pools[s]) for s in ORDER},
        "needed_by_stratum": {s: max(0, need.get(s, 0)) for s in ORDER},
        "written_by_stratum": {s: len(picked[s]) for s in ORDER},
    }
    return picked, log


def to_task(rec: dict, stratum: str) -> dict:
    aliases = [a for a in rec.get("answer_aliases") or [] if a and a != rec["answer"]]
    return {
        "id": f"musique-{rec['id']}",
        "question": rec["question"],
        "answer": rec["answer"],
        "aliases": aliases,
        "complexity": stratum,
        "source": f"musique-{len(rec['question_decomposition'])}hop",
    }


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "paragraph"


def write_corpus(selected: list[dict], corpus_dir: Path) -> dict:
    """One .md file per unique (title, text) paragraph. Replaces the directory's .md files."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for old in corpus_dir.glob("*.md"):
        old.unlink()
    seen: set[str] = set()
    total = 0
    for rec in selected:
        for p in rec["paragraphs"]:
            total += 1
            title, text = p.get("title", ""), p.get("paragraph_text", "")
            key = hashlib.sha1(f"{title}\n{text}".encode()).hexdigest()[:12]
            if key in seen:
                continue
            seen.add(key)
            (corpus_dir / f"{_slug(title)}-{key}.md").write_text(f"# {title}\n\n{text}\n")
    return {"paragraphs": total, "unique_paragraphs_written": len(seen)}


def fetch_from_mirror() -> Path:
    from huggingface_hub import hf_hub_download

    print(f"fetching {MIRROR_FILE} from the Hugging Face mirror {MIRROR_REPO}; compare its "
          "SHA-256 with the official release from StonyBrookNLP/musique", file=sys.stderr)
    return Path(hf_hub_download(MIRROR_REPO, MIRROR_FILE, repo_type="dataset"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None,
                    help="musique_ans_v1.0_dev.jsonl (default: fetch from the HF mirror)")
    ap.add_argument("--have", type=Path, nargs="*", default=[],
                    help="task files already in the suite; MuSiQue fills what they lack")
    ap.add_argument("--per-stratum", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/tasks/musique_subset.jsonl"))
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/corpus/musique"))
    args = ap.parse_args(argv)

    src = args.input or fetch_from_mirror()
    have = count_have([p for p in args.have if p.exists()])
    need = {s: args.per_stratum - have[s] for s in ORDER}
    picked, log = select(load(src), need, args.seed)
    selected = [r for s in ORDER for r in picked[s]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for s in ORDER:
            for r in picked[s]:
                fh.write(json.dumps(to_task(r, s)) + "\n")
    log.update(
        input=str(args.input) if args.input else f"hf://datasets/{MIRROR_REPO}/{MIRROR_FILE}",
        input_sha256=sha256(src),
        split="dev (MuSiQue-Ans)",
        seed=args.seed,
        have_by_stratum={s: have[s] for s in ORDER},
        corpus_dir=str(args.corpus_dir),
        **write_corpus(selected, args.corpus_dir),
    )
    Path(f"{args.out}.log.json").write_text(json.dumps(log, indent=2) + "\n")

    for s in ORDER:
        if log["written_by_stratum"][s] < log["needed_by_stratum"][s]:
            print(f"warning: only {log['written_by_stratum'][s]} eligible {s} tasks",
                  file=sys.stderr)
    print(f"MuSiQue dev: {log['rows']} rows; have {log['have_by_stratum']}; "
          f"wrote {log['written_by_stratum']} to {args.out}", file=sys.stderr)
    print(f"corpus: {log['unique_paragraphs_written']} unique paragraphs in {args.corpus_dir}; "
          f"input sha256 {log['input_sha256']}", file=sys.stderr)


if __name__ == "__main__":
    main()
