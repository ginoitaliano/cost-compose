import tempfile
from pathlib import Path

from helpers import CONFIGS, cfg

from costcompose.config import from_dict, load_config


def test_table1_configs_match_paper():
    expected = {
        "baseline": (False, False, False),
        "cache_only": (True, False, False),
        "route_only": (False, True, False),
        "composed": (True, True, True),
    }
    for name, (c, r, e) in expected.items():
        t = cfg(name).techniques
        assert (t["caching"], t["routing"], t["early_stopping"]) == (c, r, e), name


def test_paper_hyperparameters():
    c = cfg("composed")
    assert c.caching.threshold == 0.92
    assert c.early_stopping.threshold == 0.88
    assert set(c.routing.weights) == {
        "query_tokens", "tool_calls", "context_tokens", "dependency_depth"
    }


def test_all_shipped_configs_load():
    paths = list(CONFIGS.glob("*.yaml")) + list((CONFIGS / "ablations").glob("*.yaml"))
    assert len(paths) >= 9
    for p in paths:
        load_config(p)


def test_missing_price_rejected():
    raw = {"name": "x", "routing": {"default_model": "gpt-4o"}, "prices": {}}
    try:
        from_dict(raw)
    except ValueError as e:
        assert "no price entry" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_circular_extends_rejected():
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.yaml", Path(d) / "b.yaml"
        a.write_text("extends: b.yaml\nname: a\n")
        b.write_text("extends: a.yaml\nname: b\n")
        try:
            load_config(a)
        except ValueError as e:
            assert "circular" in str(e)
        else:
            raise AssertionError("expected ValueError")


def test_mode_defaults_and_validation():
    assert cfg("baseline").mode == "multi_agent"
    assert cfg("single_agent_baseline").mode == "single_agent"
    assert cfg("single_agent_route").routing.enabled
    raw = {"name": "x", "mode": "swarm",
           "prices": {"gpt-4o": {"input_per_mtok": 1, "output_per_mtok": 1},
                      "text-embedding-3-small": {"input_per_mtok": 1, "output_per_mtok": 0}}}
    try:
        from_dict(raw)
    except ValueError as e:
        assert "mode" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_every_ablation_is_in_exactly_one_make_target():
    """make run / make run-extended use explicit lists; a new ablation must be added to one."""
    import re

    text = (CONFIGS.parent / "Makefile").read_text().replace("\\\n", " ")
    lists = {}
    for name in ("CORE_CONFIGS", "EXTENDED_CONFIGS"):
        m = re.search(rf"^{name}\s*=\s*(.+)$", text, re.MULTILINE)
        lists[name] = set(re.findall(r"configs/ablations/(\w+)\.yaml", m.group(1)))
    on_disk = {p.stem for p in (CONFIGS / "ablations").glob("*.yaml")}
    assert lists["CORE_CONFIGS"] | lists["EXTENDED_CONFIGS"] == on_disk
    assert not lists["CORE_CONFIGS"] & lists["EXTENDED_CONFIGS"]


def test_parallel_parts_cover_the_core_run_exactly():
    import re

    text = (CONFIGS.parent / "Makefile").read_text().replace("\\\n", " ")

    def items(name):
        m = re.search(rf"^{name}\s*=\s*(.+)$", text, re.MULTILINE)
        return m.group(1).split()

    parts = items("PART_A") + items("PART_B") + items("PART_C")
    assert sorted(parts) == sorted(items("CORE_CONFIGS"))
    assert len(parts) == len(set(parts))
