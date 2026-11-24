import json
import tempfile
from pathlib import Path

import pytest
from helpers import CONFIGS, SAMPLE_TASKS, cfg

from costcompose import judge, spotcheck
from costcompose.cli import main
from costcompose.llm import FakeLLMClient


def _rec(config, task_id, gold, pred, success, strict, seed=0):
    return {"config": config, "seed": seed, "task_id": task_id, "complexity": "low",
            "gold": gold, "prediction": pred, "success": success, "success_strict": strict}


RECORDS = [
    _rec("baseline", "s-low-1", "95", "95 USD", True, False),     # disputed: primary accepts
    _rec("baseline", "s-low-2", "1", "1", True, True),            # agree
    _rec("route_only", "s-low-1", "95", "95", True, True),        # agree
    _rec("route_only", "s-low-2", "1", "2 hours", False, False),  # agree, wrong
]


def test_refuses_judge_on_the_routing_ladder():
    routing = cfg("route_only").routing
    for model in ("gpt-4o", "anthropic/claude-sonnet-4-5", "meta-llama/Llama-3.1-8B-Instruct",
                  "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct"):
        with pytest.raises(ValueError, match="routing ladder"):
            judge.check_off_ladder(model, routing)
    judge.check_off_ladder("hosted_vllm/Qwen/Qwen2.5-7B-Instruct", routing)  # allowed


def test_selects_only_disputed_runs_by_default():
    rows = judge.select(RECORDS)
    assert [(r["config"], r["task_id"]) for r in rows] == [("baseline", "s-low-1")]


def test_selects_spotcheck_rows_too(tmp_path):
    path = spotcheck.write_csv(spotcheck.sample(RECORDS, fraction=1.0), tmp_path / "s.csv")
    assert len(judge.select(RECORDS, path)) == len(RECORDS)


def test_parse_verdict():
    assert judge.parse_verdict("VERDICT: CORRECT") is True
    assert judge.parse_verdict("reasoning...\nverdict: incorrect") is False
    assert judge.parse_verdict("<think>VERDICT: INCORRECT</think>VERDICT: CORRECT") is True
    assert judge.parse_verdict("I think it is right") is None


def test_agreement_with_both_scores():
    judged = judge.run(RECORDS, {}, FakeLLMClient(), "fake-judge")
    res = judge.agreement(judged)
    assert res["judged"] == 4 and res["unparsed_verdicts"] == 0
    # the fake judge compares numbers: it agrees with primary on all 4, with strict on 3 of 4
    assert res["judge_vs_primary"]["agreement"] == 1.0
    assert res["judge_vs_strict"]["agreement"] == 0.75
    assert res["by_config"]["baseline"]["judge_vs_strict"]["n"] == 2


def test_cli_judge_never_touches_runs_jsonl():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        runs = "".join(json.dumps(r) + "\n" for r in RECORDS)
        (out / "runs.jsonl").write_text(runs)
        args = ["judge", str(out), "--tasks", str(SAMPLE_TASKS), "--dry-run",
                "--config", str(CONFIGS / "base.yaml")]
        assert main(args) == 0
        assert (out / "runs.jsonl").read_text() == runs
        judged = [json.loads(x) for x in (out / "judge.jsonl").read_text().splitlines()]
        assert len(judged) == 1 and judged[0]["judge_correct"] is True
        with pytest.raises(ValueError, match="routing ladder"):
            main(["judge", str(out), "--tasks", str(SAMPLE_TASKS), "--model", "gpt-4o",
                  "--config", str(CONFIGS / "base.yaml")])
