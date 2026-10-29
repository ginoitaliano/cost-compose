"""Build the GAIA portion of the task suite.

Selects GAIA validation tasks (test-split answers are hidden) without file attachments,
stratifies them by the annotators' step counts and writes JSONL in the costcompose task format.

    pip install datasets
    huggingface-cli login            # GAIA is gated: accept its terms on the Hub first
    python scripts/prepare_gaia.py --per-stratum 20 --out data/tasks/gaia_subset.jsonl

Output is git-ignored: GAIA asks that its data not be republished in crawlable form.
"""

from __future__ import annotations

import argparse
import json
import random


def stratum(steps: int, low_max: int, medium_max: int) -> str:
    if steps <= low_max:
        return "low"
    return "medium" if steps <= medium_max else "high"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=20)
    ap.add_argument("--low-max-steps", type=int, default=4)
    ap.add_argument("--medium-max-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/tasks/gaia_subset.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
    buckets: dict[str, list[dict]] = {"low": [], "medium": [], "high": []}
    for row in ds:
        if row.get("file_name"):
            continue  # attachments need modality-specific tools the pipeline does not include
        meta = row.get("Annotator Metadata") or {}
        try:
            steps = int(str(meta.get("Number of steps", "")).strip())
        except ValueError:
            continue
        cx = stratum(steps, args.low_max_steps, args.medium_max_steps)
        buckets[cx].append({
            "id": f"gaia-{row['task_id']}",
            "question": row["Question"],
            "answer": row["Final answer"],
            "complexity": cx,
            "source": f"gaia-L{row['Level']}",
        })

    rng = random.Random(args.seed)
    with open(args.out, "w") as fh:
        for cx, rows in buckets.items():
            rng.shuffle(rows)
            picked = rows[: args.per_stratum]
            if len(picked) < args.per_stratum:
                print(f"warning: only {len(picked)} {cx} tasks available")
            for r in picked:
                fh.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
