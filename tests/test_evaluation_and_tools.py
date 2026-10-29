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
