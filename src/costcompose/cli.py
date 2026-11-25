"""Command-line interface: ``costcompose run | report | validate | spotcheck | judge``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .config import load_config
from . import judge, spotcheck
from .experiment import PreflightError, run_experiment, write_report
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
    run.add_argument("--no-retry-errors", action="store_true",
                     help="on resume, keep task runs that ended in an error instead of retrying")
    run.add_argument("--skip-preflight", action="store_true",
                     help="do not call each model once before the first task")

    rep = sub.add_parser("report", help="rebuild summary.json / summary.md from runs.jsonl")
    rep.add_argument("out", type=Path)
    rep.add_argument("--exclude-tasks", type=Path, nargs="*", default=[],
                     help="task files whose task IDs are left out (sensitivity analysis)")
    rep.add_argument("--name", default=None,
                     help="output file stem (default: summary, or summary.excluded with "
                          "--exclude-tasks)")

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

    jdg = sub.add_parser("judge", help="audit the scorer with an independent LLM judge")
    jdg.add_argument("out", type=Path, help="run directory containing runs.jsonl")
    jdg.add_argument("--tasks", type=Path, required=True, help="task JSONL, for the questions")
    jdg.add_argument("--model", default=None,
                     help="LiteLLM id of a judge model NOT on the routing ladder")
    jdg.add_argument("--api-base", default=None,
                     help="endpoint for a self-served judge, e.g. http://localhost:8001/v1")
    jdg.add_argument("--include-spotcheck", type=Path, default=None,
                     help="also judge the runs listed in this spot-check CSV")
    jdg.add_argument("--config", type=Path, default=Path("configs/base.yaml"),
                     help="config whose routing ladder the judge must not be on")
    jdg.add_argument("--dry-run", action="store_true", help="offline fake judge (tests)")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    # -v is for the run's own progress. LiteLLM and httpx log every request at INFO through the
    # root logger; keep them at LITELLM_LOG (default WARNING) so -v does not print each call.
    third_party = getattr(logging, os.environ.get("LITELLM_LOG", "WARNING").upper(),
                          logging.WARNING)
    for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "httpx"):
        logging.getLogger(name).setLevel(third_party)

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

    if args.cmd == "judge":
        model = args.model or ("fake-judge" if args.dry_run else None)
        if not model:
            ap.error("judge needs --model (or --dry-run)")
        judge.check_off_ladder(model, load_config(args.config).routing)
        rows = judge.select(spotcheck.load_records(args.out), args.include_spotcheck)
        if not rows:
            print("nothing to judge: primary and strict scores agree on every run "
                  "(add --include-spotcheck to judge the spot-check sample)")
            return 0
        if args.dry_run:
            from .llm import FakeLLMClient
            client = FakeLLMClient()
        else:
            from .llm import LiteLLMClient
            client = LiteLLMClient(api_base=args.api_base)
        questions = {t.id: t.question for t in load_tasks(args.tasks)}
        result = judge.write(judge.run(rows, questions, client, model), args.out)
        print(judge.format_result(result))
        print(f"wrote {args.out / 'judge.jsonl'} and {args.out / 'judge_agreement.json'}")
        return 0

    if args.cmd == "report":
        excl = {json.loads(x)["id"] for p in args.exclude_tasks
                for x in p.read_text().splitlines() if x.strip()}
        name = args.name or ("summary.excluded" if excl else "summary")
        write_report(args.out, excl or None, name)
        print((args.out / f"{name}.md").read_text())
        return 0

    backend = "fake" if args.dry_run else args.backend
    cache_backend = args.cache_backend or ("memory" if backend == "fake" else None)
    tasks = load_tasks(args.tasks, limit=args.limit)
    try:
        run_experiment(_resolve_configs(args.configs, args.config_dir), tasks, args.corpus,
                       args.seeds, args.out, backend=backend, engine=args.engine,
                       cache_backend=cache_backend, retry_errors=not args.no_retry_errors,
                       check_models=not args.skip_preflight)
    except PreflightError as exc:
        print(exc, file=sys.stderr)
        return 2
    print((args.out / "summary.md").read_text())
    if backend == "fake":
        print("\nNote: --dry-run uses an offline fake model. These numbers check the plumbing; "
              "they are not results.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
