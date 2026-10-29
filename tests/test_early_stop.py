from costcompose.config import EarlyStoppingConfig
from costcompose.early_stop import EarlyStopper, consistency, extract_answer, parse_confidence


def test_parse_confidence():
    assert parse_confidence("ANSWER: 4\nCONFIDENCE: 0.93\nSTATUS: DRAFT") == 0.93
    assert parse_confidence("no score here") is None


def test_extract_answer():
    assert extract_answer("ANSWER: 192 GPU-hours\nCONFIDENCE: 0.9") == "192 GPU-hours"


def test_needs_two_drafts():
    s = EarlyStopper(EarlyStoppingConfig(enabled=True))
    assert s.should_stop(["ANSWER: 1\nCONFIDENCE: 1.0"]) == (False, None)


def test_stops_on_stable_confident_drafts():
    s = EarlyStopper(EarlyStoppingConfig(enabled=True, threshold=0.88))
    drafts = ["ANSWER: 750\nCONFIDENCE: 0.9", "ANSWER: 750\nCONFIDENCE: 0.95"]
    stop, conf = s.should_stop(drafts)
    assert stop and abs(conf - (0.5 * 0.95 + 0.5 * 1.0)) < 1e-9


def test_does_not_stop_when_answer_changes():
    s = EarlyStopper(EarlyStoppingConfig(enabled=True, threshold=0.88))
    drafts = ["ANSWER: 750\nCONFIDENCE: 0.95", "ANSWER: one thousand\nCONFIDENCE: 0.95"]
    assert s.should_stop(drafts)[0] is False


def test_disabled_never_stops():
    s = EarlyStopper(EarlyStoppingConfig(enabled=False))
    drafts = ["ANSWER: 1\nCONFIDENCE: 1", "ANSWER: 1\nCONFIDENCE: 1"]
    assert s.should_stop(drafts)[0] is False


def test_consistency_bounds():
    assert consistency("ANSWER: x", "ANSWER: x") == 1.0
    assert 0.0 <= consistency("ANSWER: abc", "ANSWER: xyz") < 0.5
