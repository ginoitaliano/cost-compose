"""Command-line interface: ``costcompose run | report | validate | spotcheck``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from . import spotcheck
from .experiment import run_experiment, write_report
from .tasks import load_tasks, strata

DEFAULT_CONFIGS = ["baseline", "cache_only", "route_only", "composed"]


def _resolve_configs(names: list[str], config_dir: Path) -> list[Path]:
    paths = []
    for n in names:
        p = Path(n)
        paths.append(p if p.suffix in {".yaml", ".yml"} else config_dir / f"{n}.yaml")
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="costcompose", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run configurations over a task suite")
    run.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                     help="config names (looked up in --config-dir) or YAML paths")
    run.add_argument("--config-dir", type=Path, default=Path("configs"))
    run.add_argument("--tasks", type=Path, required=True)
    run.add_argument("--corpus", type=Path, default=None)
    run.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    run.add_argument("--out", type=Path, default=Path("results"))
    run.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    run.add_argument("--backend", choices=["litellm", "fake"], default="litellm")
    run.add_argument("--dry-run", action="store_true", help="shorthand for --backend fake")
    run.add_argument("--engine", choices=["auto", "native", "langgraph"], default="auto")
    run.add_argument("--cache-backend", choices=["redis", "memory"], default=None,
                     help="override caching.backend (--dry-run defaults to memory)")

    rep = sub.add_parser("report", help="rebuild summary.json / summary.md from runs.jsonl")
    rep.add_argument("out", type=Path)

    val = sub.add_parser("validate", help="check configs and a task file without running")
    val.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    val.add_argument("--config-dir", type=Path, default=Path("configs"))
    val.add_argument("--tasks", type=Path, default=None)

    spot = sub.add_parser("spotcheck", help="human spot-checks of the automated scorer")
    spot_sub = spot.add_subparsers(dest="spot_cmd", required=True)
    smp = spot_sub.add_parser("sample", help="write a stratified sample of runs to grade")
    smp.add_argument("out", type=Path, help="run directory containing runs.jsonl")
    smp.add_argument("--tasks", type=Path, default=None,
                     help="task JSONL, to include each question in the CSV")
    smp.add_argument("--fraction", type=float, default=0.1)
    smp.add_argument("--seed", type=int, default=0)
    smp.add_argument("--csv", type=Path, default=None, help="default: <out>/spotcheck.csv")
    smp.add_argument("--force", action="store_true", help="overwrite an existing CSV")
    scr = spot_sub.add_parser("score", help="agreement of human grades with the scorer")
    scr.add_argument("csv", type=Path)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "validate":
        for p in _resolve_configs(args.configs, args.config_dir):
            cfg = load_config(p)
            on = [k for k, v in cfg.techniques.items() if v] or ["none"]
            print(f"ok  {cfg.name:<22} techniques: {', '.join(on)}")
        if args.tasks:
            tasks = load_tasks(args.tasks)
            print(f"ok  {len(tasks)} tasks, strata {strata(tasks)}")
        return 0

    if args.cmd == "spotcheck":
        if args.spot_cmd == "sample":
            questions = None
            if args.tasks:
                questions = {t.id: t.question for t in load_tasks(args.tasks)}
            rows = spotcheck.sample(spotcheck.load_records(args.out), questions,
                                    args.fraction, args.seed)
            path = spotcheck.write_csv(rows, args.csv or args.out / "spotcheck.csv", args.force)
            tasks = len({(r["task_id"], r["seed"]) for r in rows})
            print(f"wrote {path}: {len(rows)} rows ({tasks} task runs x configs). Fill in "
                  "human_success with 1 or 0, then run: costcompose spotcheck score "
                  f"{path}")
            return 0
        result = spotcheck.score(args.csv)
        out = args.csv.with_name(args.csv.stem + "_agreement.json")
        out.write_text(json.dumps(result, indent=2))
        print(spotcheck.format_score(result))
        print(f"wrote {out}")
        return 0

    if args.cmd == "report":
        write_report(args.out)
        print((args.out / "summary.md").read_text())
        return 0

    backend = "fake" if args.dry_run else args.backend
    cache_backend = args.cache_backend or ("memory" if backend == "fake" else None)
    tasks = load_tasks(args.tasks, limit=args.limit)
    run_experiment(_resolve_configs(args.configs, args.config_dir), tasks, args.corpus,
                   args.seeds, args.out, backend=backend, engine=args.engine,
                   cache_backend=cache_backend)
    print((args.out / "summary.md").read_text())
    if backend == "fake":
        print("\nNote: --dry-run uses an offline fake model. These numbers check the plumbing; "
              "they are not results.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
