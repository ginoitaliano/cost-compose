import csv
import json
import tempfile
from pathlib import Path

import pytest
from helpers import CONFIGS, SAMPLE_CORPUS, SAMPLE_TASKS

from costcompose import spotcheck
from costcompose.cli import main


def _records(per_stratum=40, seeds=(0, 1, 2), configs=("baseline", "composed")):
    out = []
    for cfg in configs:
        for cx in ("low", "medium", "high"):
            for t in range(per_stratum):
                for s in seeds:
                    out.append({"config": cfg, "seed": s, "task_id": f"{cx}{t}",
                                "complexity": cx, "gold": "1", "prediction": "1",
                                "success": True})
    return out


def test_sample_is_stratified_seeded_and_paired_across_configs():
    rows = spotcheck.sample(_records(), {"low0": "q?"}, fraction=0.1, seed=0)
    pairs = {(r["task_id"], r["seed"]) for r in rows}
    assert len(pairs) == 12  # 10% of 40 per stratum x 3 strata, one seed each
    by_cx = {cx: sum(1 for t, _ in pairs if t.startswith(cx)) for cx in ("low", "medium", "high")}
    assert by_cx == {"low": 4, "medium": 4, "high": 4}
    assert len(rows) == 24  # every config graded on the same task runs
    assert rows == spotcheck.sample(_records(), {"low0": "q?"}, fraction=0.1, seed=0)
    assert rows != spotcheck.sample(_records(), fraction=0.1, seed=1)
    assert all(r["human_success"] == "" for r in rows)


def test_small_strata_get_at_least_one_task():
    rows = spotcheck.sample(_records(per_stratum=3, seeds=(0,)), fraction=0.1)
    assert len({r["task_id"] for r in rows}) == 3


def test_score_agreement_and_kappa():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "s.csv"
        rows = []
        # 8 graded rows: auto/human = (1,1)x3, (0,0)x3, (1,0)x1, (0,1)x1; plus one ungraded
        for i, (a, h) in enumerate([(1, 1)] * 3 + [(0, 0)] * 3 + [(1, 0), (0, 1)] + [(1, None)]):
            rows.append({"config": "c" if i % 2 else "d", "seed": 0, "task_id": f"t{i}",
                         "complexity": "low", "question": "", "gold": "", "prediction": "",
                         "auto_success": a, "human_success": "" if h is None else ("yes" if h
                         else "0"), "notes": ""})
        spotcheck.write_csv(rows, path)
        with pytest.raises(FileExistsError):
            spotcheck.write_csv(rows, path)  # never clobbers grades
        res = spotcheck.score(path)
    assert res["ungraded"] == 1 and res["overall"]["n"] == 8
    assert abs(res["overall"]["agreement"] - 0.75) < 1e-12
    assert abs(res["overall"]["cohen_kappa"] - 0.5) < 1e-12  # p_e = 0.5
    assert len(res["disagreements"]) == 2
    assert sum(a["n"] for a in res["by_config"].values()) == 8


def test_cli_spotcheck_on_dry_run():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        assert main(["run", "--dry-run", "--config-dir", str(CONFIGS), "--tasks",
                     str(SAMPLE_TASKS), "--corpus", str(SAMPLE_CORPUS), "--seeds", "0", "1",
                     "--out", str(out), "--configs", "baseline", "composed"]) == 0
        assert main(["spotcheck", "sample", str(out), "--tasks", str(SAMPLE_TASKS)]) == 0
        path = out / "spotcheck.csv"
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert rows and all(r["question"] for r in rows)
        with pytest.raises(FileExistsError):
            main(["spotcheck", "sample", str(out)])
        for r in rows:
            r["human_success"] = r["auto_success"]
        spotcheck.write_csv(rows, path, force=True)
        assert main(["spotcheck", "score", str(path)]) == 0
        res = json.loads((out / "spotcheck_agreement.json").read_text())
        assert res["overall"]["agreement"] == 1.0 and res["overall"]["n"] == len(rows)
