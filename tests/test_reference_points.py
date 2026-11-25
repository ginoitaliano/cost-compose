"""Answer F1 and the single-model oracle bound."""

from helpers import cfg

from costcompose.analysis import _oracle
from costcompose.evaluation import answer_f1
from costcompose.experiment import models_used


def test_answer_f1_gives_partial_credit_and_takes_the_best_alias():
    assert answer_f1("1999", ["30 November 1999"]) == 0.5
    assert answer_f1("Sarmatians, Goths", ["Iranic Sarmatians and the Germanic Goths",
                                          "Sarmatians"]) == 2 / 3
    assert answer_f1("The Premium plan.", ["premium plan"]) == 1.0
    assert answer_f1("Unknown", ["Pope John Paul II", "John Paul II"]) == 0.0
    assert answer_f1("", ["x"]) == 0.0


def test_single_model_configs_use_one_model_each():
    assert models_used(cfg("llama_only")) == ["hosted_vllm/meta-llama/Llama-3.1-8B-Instruct"]
    assert models_used(cfg("haiku_only")) == ["anthropic/claude-haiku-4-5"]
    assert models_used(cfg("sonnet_only")) == ["anthropic/claude-sonnet-4-5"]


def _rec(config, task, success, cost, cx="low"):
    return {"config": config, "task_id": task, "seed": 0, "success": success, "cost_usd": cost,
            "complexity": cx}


def test_oracle_takes_the_cheapest_model_that_succeeds():
    costs = {"llama_only": 0.001, "haiku_only": 0.004, "baseline": 0.02, "sonnet_only": 0.03}
    solves = {  # task -> configs that solve it
        "t1": {"llama_only", "haiku_only", "baseline", "sonnet_only"},
        "t2": {"baseline", "sonnet_only"},
        "t3": {"sonnet_only"},
        "t4": set(),
    }
    recs = [_rec(c, t, c in ok, costs[c]) for t, ok in solves.items() for c in costs]
    recs += [_rec("route_only", t, False, 0.01) for t in solves]
    configs = {c: {} for c in [*costs, "route_only"]}
    o = _oracle(recs, configs)
    assert o["picks"]["llama_only"]["rate"] == 0.25 and o["picks"]["baseline"]["rate"] == 0.25
    assert o["picks"]["sonnet_only"]["rate"] == 0.25 and o["picks"]["none"]["rate"] == 0.25
    assert o["oracle_success"]["rate"] == 0.75
    assert abs(o["oracle_cost_usd"] - (0.001 + 0.02 + 0.03 + 0.001) / 4) < 1e-12
    assert abs(o["oracle_saving"] - (1 - 0.052 / 0.08)) < 1e-9
    assert abs(o["routed"]["route_only"]["saving"] - 0.5) < 1e-9


def test_oracle_needs_baseline_and_another_model():
    recs = [_rec("baseline", "t", True, 0.02)]
    assert _oracle(recs, {"baseline": {}}) is None
