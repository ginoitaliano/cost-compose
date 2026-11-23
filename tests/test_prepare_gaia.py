"""GAIA filter, tested on synthetic rows shaped like GAIA records (no dataset download)."""

import argparse
import importlib.util

from helpers import ROOT

_spec = importlib.util.spec_from_file_location("prepare_gaia", ROOT / "scripts" / "prepare_gaia.py")
gaia = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gaia)


def _row(i, tools="1. Calculator", file_name="", question="What is 2 + 2?", steps="3"):
    return {"task_id": f"t{i}", "Question": question, "Final answer": "4", "Level": 1,
            "file_name": file_name,
            "Annotator Metadata": {"Tools": tools, "Number of steps": steps}}


def _args(**kw):
    base = dict(low_max_steps=4, medium_max_steps=8, readable_extensions=[])
    return argparse.Namespace(**{**base, **kw})


def test_exclusion_reasons():
    assert gaia.exclusion_reasons(_row(0)) == []
    assert gaia.exclusion_reasons(_row(0, tools="1. Web browser\n2. Search engine")) == [
        "web_browsing"]
    assert gaia.exclusion_reasons(_row(0, question="See https://example.org/x")) == [
        "web_browsing"]
    assert gaia.exclusion_reasons(_row(0, file_name="a.png")) == ["multimodal_input"]
    assert gaia.exclusion_reasons(_row(0, tools="1. Video processing software")) == [
        "multimodal_input"]
    assert gaia.exclusion_reasons(_row(0, file_name="sheet.xlsx")) == ["unreadable_attachment"]
    both = gaia.exclusion_reasons(_row(0, file_name="clip.mp3", tools="1. Web browser"))
    assert both == ["multimodal_input", "web_browsing"]


def test_readable_extensions_allow_text_attachments():
    assert gaia.exclusion_reasons(_row(0, file_name="notes.txt")) == ["unreadable_attachment"]
    assert gaia.exclusion_reasons(_row(0, file_name="notes.txt"), frozenset({".txt"})) == []


def test_select_counts_exclusions_by_reason_and_stratifies():
    rows = [
        _row(0, steps="2"),
        _row(1, steps="6"),
        _row(2, steps="12"),
        _row(3, tools="1. Web browser"),
        _row(4, file_name="a.jpg", tools="1. Web browser"),
        _row(5, file_name="data.xlsx"),
        _row(6, steps="n/a"),
        _row(7, tools=None, steps="1"),
    ]
    buckets, log = gaia.select(rows, _args())
    assert log["split"] == "validation" and log["rows"] == 8
    assert log["excluded_by_primary_reason"] == {
        "web_browsing": 1, "multimodal_input": 1, "unreadable_attachment": 1, "no_step_count": 1}
    assert log["excluded_by_any_reason"]["web_browsing"] == 2
    assert log["excluded"] == 4
    assert log["eligible_by_stratum"] == {"low": 2, "medium": 1, "high": 1}
    assert log["eligible_without_tool_metadata"] == 1
    assert {r["complexity"] for r in buckets["high"]} == {"high"}
