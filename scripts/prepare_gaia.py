"""Build the GAIA portion of the task suite.

Uses the GAIA validation split only: its answers are public, while test-split answers are
hidden. Tasks the pipeline cannot attempt are excluded, and the exclusions are logged by
reason. The pipeline's tools are a calculator and search over the local --corpus, so a task is
excluded when it:

* needs web browsing (annotator tools such as a web browser or search engine, or a URL in the
  question);
* comes with an attached file the pipeline cannot read (any attachment by default; the
  pipeline has no per-task file input);
* needs multimodal input (image, audio or video attachments, or annotator tools such as image
  recognition, speech-to-text or video processing).

Exclusion is based on GAIA's own metadata: ``file_name`` and the annotators' tool list
(``Annotator Metadata`` -> ``Tools``). Tasks whose tool list is missing are kept and counted
separately so they can be reviewed. Remaining tasks are stratified by the annotators' step
counts and written as JSONL in the costcompose task format.

    pip install datasets
    huggingface-cli login            # GAIA is gated: accept its terms on the Hub first
    python scripts/prepare_gaia.py --per-stratum 20 --out data/tasks/gaia_subset.jsonl

Output is git-ignored: GAIA asks that its data not be republished in crawlable form. The
exclusion counts (no task content) are written next to it as ``<out>.exclusions.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

SPLIT = "validation"  # the only GAIA split with public answers

WEB_TOOLS = (
    "web browser", "browser", "search engine", "google", "bing", "internet", "web search",
    "wikipedia", "website", "web page", "webpage", "online",
)
MULTIMODAL_TOOLS = (
    "image recognition", "image", "video", "audio", "speech", "transcri", "ocr", "vision",
    "youtube", "color", "colour",
)
MULTIMODAL_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".svg",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".mov", ".avi", ".webm",
}
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)

# Order used to pick one primary reason per excluded task (counts then sum to the total).
REASONS = ("multimodal_input", "unreadable_attachment", "web_browsing")


def tool_list(row: dict) -> str | None:
    meta = row.get("Annotator Metadata") or {}
    tools = meta.get("Tools")
    return None if tools is None or not str(tools).strip() else str(tools).lower()


def exclusion_reasons(row: dict, readable_extensions: frozenset[str] = frozenset()) -> list[str]:
    """All reasons a GAIA row is out of scope for the pipeline, in REASONS order."""
    reasons: set[str] = set()
    tools = tool_list(row) or ""
    file_name = (row.get("file_name") or "").strip()
    if file_name:
        ext = Path(file_name).suffix.lower()
        if ext in MULTIMODAL_EXTENSIONS:
            reasons.add("multimodal_input")
        elif ext not in readable_extensions:
            reasons.add("unreadable_attachment")
    if any(k in tools for k in MULTIMODAL_TOOLS) or "youtube" in row.get("Question", "").lower():
        reasons.add("multimodal_input")
    if any(k in tools for k in WEB_TOOLS) or _URL.search(row.get("Question", "")):
        reasons.add("web_browsing")
    return [r for r in REASONS if r in reasons]


def stratum(steps: int, low_max: int, medium_max: int) -> str:
    if steps <= low_max:
        return "low"
    return "medium" if steps <= medium_max else "high"


def step_count(row: dict) -> int | None:
    meta = row.get("Annotator Metadata") or {}
    try:
        return int(str(meta.get("Number of steps", "")).strip())
    except ValueError:
        return None


def select(rows, args) -> tuple[dict[str, list[dict]], dict]:
    readable = frozenset(e.lower() for e in args.readable_extensions)
    buckets: dict[str, list[dict]] = {"low": [], "medium": [], "high": []}
    primary: Counter = Counter()
    any_reason: Counter = Counter()
    total = no_tools = 0
    for row in rows:
        total += 1
        reasons = exclusion_reasons(row, readable)
        if reasons:
            primary[reasons[0]] += 1
            any_reason.update(reasons)
            continue
        steps = step_count(row)
        if steps is None:
            primary["no_step_count"] += 1
            any_reason["no_step_count"] += 1
            continue
        if tool_list(row) is None:
            no_tools += 1
        cx = stratum(steps, args.low_max_steps, args.medium_max_steps)
        buckets[cx].append({
            "id": f"gaia-{row['task_id']}",
            "question": row["Question"],
            "answer": row["Final answer"],
            "complexity": cx,
            "source": f"gaia-L{row['Level']}",
        })
    log = {
        "split": SPLIT,
        "rows": total,
        "excluded": sum(primary.values()),
        "excluded_by_primary_reason": dict(primary),
        "excluded_by_any_reason": dict(any_reason),
        "eligible_by_stratum": {k: len(v) for k, v in buckets.items()},
        "eligible_without_tool_metadata": no_tools,
    }
    return buckets, log


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=20)
    ap.add_argument("--low-max-steps", type=int, default=4)
    ap.add_argument("--medium-max-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--readable-extensions", nargs="*", default=[],
                    help="attachment extensions the pipeline can read (default: none)")
    ap.add_argument("--out", default="data/tasks/gaia_subset.jsonl")
    args = ap.parse_args(argv)

    from datasets import load_dataset

    ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split=SPLIT)
    buckets, log = select(ds, args)

    rng = random.Random(args.seed)
    written: dict[str, int] = {}
    with open(args.out, "w") as fh:
        for cx, rows in buckets.items():
            rng.shuffle(rows)
            picked = rows[: args.per_stratum]
            written[cx] = len(picked)
            if len(picked) < args.per_stratum:
                print(f"warning: only {len(picked)} eligible {cx} tasks", file=sys.stderr)
            for r in picked:
                fh.write(json.dumps(r) + "\n")
    log["written_by_stratum"] = written

    print(f"GAIA {SPLIT} split: {log['rows']} rows, {log['excluded']} excluded", file=sys.stderr)
    for reason, n in sorted(log["excluded_by_primary_reason"].items()):
        print(f"  excluded ({reason}): {n}", file=sys.stderr)
    if log["eligible_without_tool_metadata"]:
        print(f"  kept without annotator tool list: {log['eligible_without_tool_metadata']}",
              file=sys.stderr)
    Path(f"{args.out}.exclusions.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"wrote {args.out} ({sum(written.values())} tasks) and {args.out}.exclusions.json")


if __name__ == "__main__":
    main()
