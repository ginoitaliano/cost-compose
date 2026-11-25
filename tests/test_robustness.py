"""Run-robustness: tool-call history shape, retrying errored runs, model preflight."""

import json
import tempfile
from pathlib import Path

import pytest
from helpers import CONFIGS, SAMPLE_CORPUS, SAMPLE_TASKS, cfg
from test_pipeline import _runtime

from costcompose.agents import reason_node
from costcompose.cli import main
from costcompose.experiment import PreflightError, preflight
from costcompose.llm import LLMResponse, ToolCall, Usage
from costcompose.tracking import CallRecord

STATE = {"question": "q", "subtasks": ["s"], "index": 0, "outputs": [], "context": ""}


def test_parallel_tool_calls_are_written_one_per_assistant_turn():
    """A model asking for two tools at once must not leave a two-call turn in the history:
    Llama 3.1's chat template rejects that, and routing can send the next step to Llama."""
    rt = _runtime("composed")
    seen = []
    script = iter([
        LLMResponse("", "gpt-4o", Usage(1, 1), [
            ToolCall("c1", "calculator", {"expression": "2*3"}),
            ToolCall("c2", "calculator", {"expression": "4*5"}),
        ]),
        LLMResponse("ANSWER: 26\nCONFIDENCE: 0.9\nSTATUS: FINAL", "gpt-4o", Usage(1, 1)),
    ])

    def call_llm(agent, messages, features, local_input, pipeline_context, use_tools=False,
                 **_):
        seen.append([dict(m) for m in messages])
        rt.trace.calls.append(CallRecord(agent, "gpt-4o", "fixed", 0.0, 0, 0, 0.0, 0.0, False,
                                         False, 0, 0.0))
        return next(script)

    rt.call_llm = call_llm
    out = reason_node(rt, STATE)
    assert out["outputs"] == ["26"]
    history = seen[1][2:]  # after system + user
    assert [m["role"] for m in history] == ["assistant", "tool", "assistant", "tool"]
    assert all(len(m["tool_calls"]) == 1 for m in history if m["role"] == "assistant")
    assert [m["tool_call_id"] for m in history if m["role"] == "tool"] == ["c1", "c2"]
    assert [m["content"] for m in history if m["role"] == "tool"] == ["6", "20"]
    assert rt.trace.tool_invocations == 2


def test_resume_retries_errored_runs_and_keeps_them():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        args = ["run", "--dry-run", "--config-dir", str(CONFIGS), "--tasks", str(SAMPLE_TASKS),
                "--corpus", str(SAMPLE_CORPUS), "--seeds", "0", "--out", str(out),
                "--configs", "baseline"]
        assert main(args) == 0
        runs = out / "runs.jsonl"
        rows = [json.loads(x) for x in runs.read_text().splitlines()]
        rows[0]["error"] = "NotFoundError: endpoint down"
        runs.write_text("".join(json.dumps(r) + "\n" for r in rows))

        assert main(args) == 0
        after = [json.loads(x) for x in runs.read_text().splitlines()]
        keys = [(r["config"], r["seed"], r["task_id"]) for r in after]
        assert len(after) == len(rows) and len(set(keys)) == len(keys)
        assert not any(r["error"] for r in after)
        kept = [json.loads(x) for x in (out / "runs.errors.jsonl").read_text().splitlines()]
        assert [r["task_id"] for r in kept] == [rows[0]["task_id"]]

        rows = after
        rows[0]["error"] = "still down"
        runs.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert main([*args, "--no-retry-errors"]) == 0
        assert sum(1 for x in runs.read_text().splitlines() if json.loads(x)["error"]) == 1


class _Client:
    def __init__(self, broken):
        self.broken, self.calls = broken, []

    def complete(self, model, messages, tools=None, temperature=0.0, seed=None, **_):
        self.calls.append(model)
        if model in self.broken:
            raise RuntimeError("404 not found")
        return LLMResponse("OK", model, Usage(1, 1))


class _Embedder:
    def embed(self, texts):
        return None, 1


def test_preflight_checks_every_model_and_names_failures(monkeypatch):
    configs = [cfg("baseline"), cfg("route_only")]
    client = _Client(set())
    models = preflight(configs, client, {"text-embedding-3-small": _Embedder()})
    assert set(models) == {"gpt-4o", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-5",
                           "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct"}
    assert len(client.calls) == 4  # each model once, however many configs use it

    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    broken = _Client({"hosted_vllm/meta-llama/Llama-3.1-8B-Instruct"})
    with pytest.raises(PreflightError, match="HOSTED_VLLM_API_BASE is not set"):
        preflight(configs, broken, {"text-embedding-3-small": _Embedder()})


def test_verbose_does_not_turn_on_litellm_request_logging(monkeypatch):
    import logging

    monkeypatch.delenv("LITELLM_LOG", raising=False)
    with tempfile.TemporaryDirectory() as d:
        assert main(["-v", "run", "--dry-run", "--config-dir", str(CONFIGS), "--tasks",
                     str(SAMPLE_TASKS), "--corpus", str(SAMPLE_CORPUS), "--seeds", "0",
                     "--out", d, "--configs", "baseline"]) == 0
    assert logging.getLogger("LiteLLM").level == logging.WARNING
    assert logging.getLogger("httpx").level == logging.WARNING


def test_every_request_carries_the_output_cap():
    rt = _runtime("route_only")
    seen = []

    class Capture:
        def complete(self, model, messages, tools=None, temperature=0.0, seed=None,
                     max_tokens=None):
            seen.append(max_tokens)
            return LLMResponse("ANSWER: 1\nCONFIDENCE: 0.9\nSTATUS: FINAL", model, Usage(1, 1))

    rt.llm = Capture()
    reason_node(rt, dict(STATE))
    assert seen and set(seen) == {4096}
    assert cfg("baseline").agents.max_output_tokens == 4096
