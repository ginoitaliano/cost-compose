from costcompose.evaluation import extract_final_answer, score
from costcompose.tools import ToolRegistry, safe_eval


def test_numeric_match_tolerates_formatting():
    assert score("FINAL ANSWER: 2,457.60", "2457.6")
    assert score("FINAL ANSWER: $750", "750")
    assert not score("FINAL ANSWER: 751", "750")


def test_text_match_normalises():
    assert score("FINAL ANSWER: The Premium plan.", "premium plan")


def test_list_match():
    assert score("FINAL ANSWER: a, b, c", "a, b, c")
    assert not score("FINAL ANSWER: a, b", "a, b, c")


def test_extract_handles_empty():
    assert extract_final_answer("") == ""
    assert extract_final_answer("FINAL ANSWER:   \n") == ""


def test_safe_eval():
    assert safe_eval("192 * 3.2 * 4") == 2457.6000000000004
    for bad in ("__import__('os')", "a + 1", "2 ** 1000", "(1).real"):
        try:
            safe_eval(bad)
        except (ValueError, SyntaxError):
            pass
        else:
            raise AssertionError(bad)


def test_tool_errors_are_returned_not_raised():
    reg = ToolRegistry()
    assert reg.call("calculator", {"expression": "4*250"}) == "1000"
    assert reg.call("calculator", {"expression": "nope"}).startswith("error")
    assert reg.call("missing", {}).startswith("error")


# -- primary (lenient) vs strict scoring --------------------------------------------------

import json  # noqa: E402

from helpers import SAMPLE_TASKS  # noqa: E402

from costcompose.evaluation import score_lenient  # noqa: E402

_TASKS = {json.loads(x)["id"]: json.loads(x) for x in SAMPLE_TASKS.read_text().splitlines()}

# The 18 predictions from the first real-model smoke test (baseline = GPT-4o, route_only =
# mostly Llama-3.1-8B and Claude Haiku 4.5), with the verdict each rule must give.
SMOKE = [
    # task, prediction, strict, primary
    ("s-high-2", "1475 USD", False, True),
    ("s-med-3", "4000 USD", False, True),
    ("s-low-2", "1 hour", False, True),
    ("s-med-1", "192", True, True),
    ("s-med-2", "750 USD", False, True),
    ("s-low-3", "64 GPU-hours per week", False, True),
    ("s-low-1", "95 USD", False, True),
    ("s-high-3", "10000 USD", False, True),
    ("s-high-1", "2457.6 USD", False, True),
    ("s-high-3", "4000", False, False),  # a real error stays an error
    ("s-high-1", "2457.60", True, True),
]


def _both(task_id, prediction):
    t = _TASKS[task_id]
    out = f"FINAL ANSWER: {prediction}"
    return score(out, t["answer"]), score_lenient(out, t["answer"], t["question"])


def test_smoke_predictions_scored_as_expected():
    for task_id, pred, strict, primary in SMOKE:
        assert _both(task_id, pred) == (strict, primary), (task_id, pred)


def test_lenient_only_accepts_units_the_question_names():
    for task_id, pred in [
        ("s-low-2", "1 day"),             # question asks in hours
        ("s-low-1", "95 EUR"),            # question asks in USD
        ("s-low-1", "95 thousand USD"),   # scale word changes the value
        ("s-low-1", "approx. 95 USD"),    # hedge before the number
        ("s-low-1", "95 USD (approx)"),   # hedge after it
        ("s-low-1", "95 or 100 USD"),     # two values
        ("s-low-1", "96 USD"),            # wrong number
    ]:
        assert _both(task_id, pred) == (False, False), (task_id, pred)
    assert _both("s-low-1", "$95") == (True, True)
    assert _both("s-high-1", "2,457.60 USD") == (False, True)


def test_lenient_matches_strict_on_text_answers():
    assert score_lenient("FINAL ANSWER: The Premium plan.", "premium plan", "Which plan?")
    assert not score_lenient("FINAL ANSWER: Premium plan USD", "premium plan", "in USD?")
    assert score_lenient("FINAL ANSWER: 3 USD, 4 USD", "3, 4", "Both prices in USD?")


def test_aliases_count_for_both_scores(tmp_path):
    from costcompose.tasks import load_tasks

    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"id": "m1", "question": "Who founded the company?",
                             "answer": "William Henry Gates III", "complexity": "low",
                             "aliases": ["Bill Gates", "Gates"]}) + "\n")
    t = load_tasks(p)[0]
    assert t.answers == ("William Henry Gates III", "Bill Gates", "Gates")
    out = "FINAL ANSWER: Bill Gates"
    assert any(score(out, a) for a in t.answers)
    assert any(score_lenient(out, a, t.question) for a in t.answers)
    assert not any(score("FINAL ANSWER: Paul Allen", a) for a in t.answers)
    assert load_tasks(SAMPLE_TASKS)[0].aliases == ()
