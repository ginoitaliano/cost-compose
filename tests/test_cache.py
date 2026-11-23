import importlib.util
import os

import pytest

from costcompose.cache import (
    InMemoryBackend,
    SemanticCache,
    extract_entities,
    resolve_version_key,
)
from costcompose.llm import HashingEmbedder, LLMResponse, ToolCall, Usage


def _cache(**kw):
    return SemanticCache(InMemoryBackend(), HashingEmbedder(), **kw)


def _resp(text, model="m1"):
    return LLMResponse(text, model, Usage(10, 5))


def test_exact_repeat_hits():
    c = _cache(threshold=0.92)
    hit, vec, _ = c.lookup("reasoner", "m1", "what is the meal allowance", "")
    assert hit is None
    c.store("reasoner", "m1", vec, _resp("95"))
    hit, _, _ = c.lookup("reasoner", "m1", "what is the meal allowance", "")
    assert hit is not None and hit.response.content == "95"
    assert c.stats.hit_rate("reasoner") == 0.5


def test_dissimilar_misses():
    c = _cache(threshold=0.92)
    _, vec, _ = c.lookup("reasoner", "m1", "what is the meal allowance", "")
    c.store("reasoner", "m1", vec, _resp("95"))
    hit, _, _ = c.lookup("reasoner", "m1", "how many gpu hours per week for charter teams", "")
    assert hit is None


def test_agent_scope_isolates_agents():
    c = _cache(scope="agent")
    _, vec, _ = c.lookup("retriever", "m1", "same input", "")
    c.store("retriever", "m1", vec, _resp("r"))
    hit, _, _ = c.lookup("reasoner", "m1", "same input", "")
    assert hit is None


def test_pipeline_scope_sensitive_to_upstream_context():
    c = _cache(scope="pipeline")
    _, vec, _ = c.lookup("reasoner", "m1", "subtask", "earlier: alpha beta gamma")
    c.store("reasoner", "m1", vec, _resp("x"))
    same, _, _ = c.lookup("reasoner", "m1", "subtask", "earlier: alpha beta gamma")
    drifted, _, _ = c.lookup("reasoner", "m1", "subtask",
                             "earlier: totally different upstream outputs here now")
    assert same is not None
    assert drifted is None


def test_cross_model_hits_counted_and_model_aware_blocks_them():
    c = _cache(model_aware=False)
    _, vec, _ = c.lookup("reasoner", "big", "q", "")
    c.store("reasoner", "big", vec, _resp("a", "big"))
    hit, _, _ = c.lookup("reasoner", "small", "q", "")
    assert hit is not None and hit.source_model == "big"
    assert c.stats.cross_model_hits["reasoner"] == 1

    strict = _cache(model_aware=True)
    _, vec, _ = strict.lookup("reasoner", "big", "q", "")
    strict.store("reasoner", "big", vec, _resp("a", "big"))
    hit, _, _ = strict.lookup("reasoner", "small", "q", "")
    assert hit is None


def test_reset_clears_entries_and_stats():
    c = _cache()
    _, vec, _ = c.lookup("a", "m", "q", "")
    c.store("a", "m", vec, _resp("x"))
    c.reset()
    hit, _, _ = c.lookup("a", "m", "q", "")
    assert hit is None and c.stats.hit_rate() == 0.0


# -- entity extraction and false-hit accounting --------------------------------------------

ORDER_Q = ("What is the shipping status and expected delivery window for customer order "
           "ORD-{} placed through the enterprise portal last week?")


def test_extract_entities_ids_numbers_dates():
    ents = extract_entities("Invoice #5521 for 2,457.60 USD due 2026-03-14, ref INV_77, "
                            "on March 3rd, 2026")
    assert {"id:#5521", "id:inv_77", "num:2457.6", "date:2026-03-14",
            "date:march 3rd, 2026"} <= ents
    assert extract_entities("no entities here at all") == frozenset()
    assert extract_entities("ANSWER: 750\nCONFIDENCE: 0.93\nSTATUS: DRAFT") == {"num:750"}
    # formatting differences are not different entities
    assert extract_entities("total 2,457.60") == extract_entities("total 2457.6")


def test_order_id_false_hit_flagged_under_hashing_embedder():
    """Constructed case: two requests that differ only in an order ID clear the 0.92
    threshold under the offline embedder. The hit is served but flagged as a mismatch."""
    c = _cache(threshold=0.92)
    a, b = ORDER_Q.format(48213), ORDER_Q.format(51907)
    vecs, _ = c.embedder.embed([a, b])
    assert float(vecs[0] @ vecs[1]) >= 0.92
    _, vec, _ = c.lookup("reasoner", "m1", a, "")
    c.store("reasoner", "m1", vec, _resp("ANSWER: shipped on 2026-03-02"), a)
    hit, _, _ = c.lookup("reasoner", "m1", b, "")
    assert hit is not None and hit.entity_mismatch
    assert c.stats.entity_mismatch_hits["reasoner"] == 1


def test_exact_repeat_is_not_an_entity_mismatch():
    c = _cache(threshold=0.92)
    q = ORDER_Q.format(48213)
    _, vec, _ = c.lookup("reasoner", "m1", q, "")
    c.store("reasoner", "m1", vec, _resp("x"), q)
    hit, _, _ = c.lookup("reasoner", "m1", q, "")
    assert hit is not None and not hit.entity_mismatch


def test_entries_without_entities_are_not_counted_as_mismatch():
    c = _cache(threshold=0.92)
    _, vec, _ = c.lookup("reasoner", "m1", ORDER_Q.format(1), "")
    c.store("reasoner", "m1", vec, _resp("x"))  # no local_input: entities unknown
    hit, _, _ = c.lookup("reasoner", "m1", ORDER_Q.format(2), "")
    assert hit is not None and not hit.entity_mismatch


@pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") and importlib.util.find_spec("litellm")),
    reason="needs OPENAI_API_KEY and litellm for the real text-embedding-3-small embedder",
)
def test_order_id_false_hit_under_real_embedder():
    """Same pair under text-embedding-3-small (the paper's embedder) at the 0.92 threshold."""
    from costcompose.llm import LiteLLMEmbedder

    c = SemanticCache(InMemoryBackend(), LiteLLMEmbedder("text-embedding-3-small"),
                      threshold=0.92)
    a, b = ORDER_Q.format(48213), ORDER_Q.format(51907)
    vecs, _ = c.embedder.embed([a, b])
    assert float(vecs[0] @ vecs[1]) >= 0.92
    _, vec, _ = c.lookup("reasoner", "m1", a, "")
    c.store("reasoner", "m1", vec, _resp("x"), a)
    hit, _, _ = c.lookup("reasoner", "m1", b, "")
    assert hit is not None and hit.entity_mismatch


# -- skip_tool_calls -------------------------------------------------------------------------


def _tool_resp(args):
    return LLMResponse("", "m1", Usage(10, 5), [ToolCall("c1", "calculator", args)])


def test_skip_tool_calls_withholds_action_calls_from_store():
    c = _cache(skip_tool_calls=True)
    _, vec, _ = c.lookup("reasoner", "m1", "compute 192 * 3.2", "")
    assert c.store("reasoner", "m1", vec, _tool_resp({"expression": "192 * 3.2"})) is False
    hit, _, _ = c.lookup("reasoner", "m1", "compute 192 * 3.2", "")
    assert hit is None
    # final text and argument-free calls are still cached
    assert c.store("reasoner", "m1", vec, _resp("ANSWER: 614.4")) is True
    assert c.store("orchestrator", "m1", vec, _tool_resp({})) is True


def test_skip_tool_calls_refuses_to_serve_existing_action_entries():
    backend = InMemoryBackend()
    loose = SemanticCache(backend, HashingEmbedder())
    _, vec, _ = loose.lookup("reasoner", "m1", "compute 192 * 3.2", "")
    assert loose.store("reasoner", "m1", vec, _tool_resp({"expression": "192 * 3.2"}))
    hit, _, _ = loose.lookup("reasoner", "m1", "compute 192 * 3.2", "")
    assert hit is not None and hit.response.tool_calls  # default: action calls are served

    strict = SemanticCache(backend, HashingEmbedder(), skip_tool_calls=True)
    hit, _, _ = strict.lookup("reasoner", "m1", "compute 192 * 3.2", "")
    assert hit is None
    assert strict.stats.tool_call_entries_refused["reasoner"] == 1


# -- version_key ----------------------------------------------------------------------------


@pytest.mark.parametrize("scope", ["agent", "pipeline"])
def test_version_change_produces_a_miss(scope):
    backend = InMemoryBackend()
    v1 = SemanticCache(backend, HashingEmbedder(), scope=scope, version_key="v1")
    _, vec, _ = v1.lookup("reasoner", "m1", "what is the meal allowance", "ctx")
    v1.store("reasoner", "m1", vec, _resp("95"))
    hit, _, _ = v1.lookup("reasoner", "m1", "what is the meal allowance", "ctx")
    assert hit is not None

    v2 = SemanticCache(backend, HashingEmbedder(), scope=scope, version_key="v2")
    hit, _, _ = v2.lookup("reasoner", "m1", "what is the meal allowance", "ctx")
    assert hit is None  # same backend, same request, new instructions: miss

    unversioned = SemanticCache(backend, HashingEmbedder(), scope=scope)
    hit, _, _ = unversioned.lookup("reasoner", "m1", "what is the meal allowance", "ctx")
    assert hit is None


def test_resolve_version_key(tmp_path):
    assert resolve_version_key(None) is None
    assert resolve_version_key("v3") == "v3"
    assert resolve_version_key(2) == "2"
    p = resolve_version_key("prompts")
    assert p.startswith("prompts-") and p == resolve_version_key("prompts")
    f = tmp_path / "instructions.md"
    f.write_text("Answer in one word.")
    k1 = resolve_version_key(f"file:{f}")
    f.write_text("Answer in one sentence.")
    assert resolve_version_key(f"file:{f}") != k1


def test_unversioned_namespaces_are_unchanged():
    c = _cache(scope="agent")
    assert c._namespace("reasoner", "m1") == "agent:reasoner"
    assert _cache(scope="pipeline", model_aware=True)._namespace("x", "m") == "pipeline|model:m"
