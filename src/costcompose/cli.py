"""Command-line interface: ``costcompose run | report | validate``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
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
