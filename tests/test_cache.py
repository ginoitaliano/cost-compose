from costcompose.cache import InMemoryBackend, SemanticCache
from costcompose.llm import HashingEmbedder, LLMResponse, Usage


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
