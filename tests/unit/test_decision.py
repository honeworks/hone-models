import json

import pytest
import respx

import hone_models as mk
from hone_models.decision.emulate import decision_messages, normalize, yes_probabilities
from hone_models.decision.questions import check_questions
from hone_models.errors import ConfigError, ProviderError
from hone_models.providers.jev import from_response

URL = "http://127.0.0.1:11434"
YES_NO = {"type": "yes_no", "instructions": "Ok?"}
SCORE = {"type": "score", "instructions": "How good?", "scale": [0, 10]}


@pytest.mark.parametrize(
    ("questions", "message"),
    [
        ({}, "at least one question"),
        ({"q": {"type": "rank", "instructions": "x"}}, "type must be one of"),
        ({"q": {"type": "yes_no"}}, "'instructions' is required"),
        ({"q": {"type": "choice", "instructions": "x", "options": ["a"]}}, "at least two"),
        ({"q": {"type": "score", "instructions": "x", "scale": [5, 1]}}, "low < high"),
    ],
)
def test_invalid_questions(questions, message) -> None:
    with pytest.raises(ConfigError, match=message):
        check_questions(questions)


def test_score_scale_defaults_and_mapping_questions_are_copied() -> None:
    q = {"type": "score", "instructions": "x"}
    assert check_questions({"s": q})["s"]["scale"] == [1, 5]
    assert "scale" not in q


@pytest.mark.parametrize(
    ("q", "item", "error"),
    [
        (YES_NO, None, "no answer for this question"),
        (YES_NO, {"answer": "maybe", "confidence": 0.5}, "expected yes or no"),
        (YES_NO, {"answer": "yes", "confidence": True}, "confidence must be a number"),
        (YES_NO, {"answer": "yes", "confidence": 1.5}, "not between 0 and 1"),
        (YES_NO, {"answer": "yes"}, "confidence must be a number"),
        (SCORE, {"answer": "high", "confidence": 0.5}, "a score must be a number"),
        (SCORE, {"answer": 11, "confidence": 0.5}, "outside the scale"),
    ],
)
def test_normalize_errors_are_per_answer(q, item, error) -> None:
    a = normalize(q, item)
    assert a["value"] is None
    assert error in a["error"]
    assert a["type"] == q["type"]


def test_normalize_no_and_zero_score_are_values_not_errors() -> None:
    no = normalize(YES_NO, {"answer": "No", "confidence": 0.9, "rationale": None})
    assert no["value"] == pytest.approx(0.1)
    assert no["rationale"] is None
    zero = normalize(SCORE, {"answer": 0, "confidence": 1.0, "rationale": "bad"})
    assert (zero["value"], zero["raw"], zero["error"]) == (0.0, 0.0, None)


def tokens(*pairs):
    return [{"token": t, "logprob": -0.1, "top": top or {t: -0.1}} for t, top in pairs]


def test_yes_probabilities_alignment_and_fallbacks() -> None:
    text = '{"a": {"answer": "no"}, "b": {"answer": "yes"}}'
    lp = tokens(
        ('{"a": {"answer": ', None),
        ('"no', {'"no': -0.2, '"yes': -1.8}),
        ('"}, "b": {"answer": "', None),
        ("yes", {"yes": -0.1, "maybe": -3.0}),
        ('"}}', None),
    )
    got = yes_probabilities(text, lp, ["a", "b", "missing"])
    assert got["a"] == pytest.approx(0.168, abs=0.001)
    assert got["b"] == 1.0
    assert "missing" not in got
    assert yes_probabilities(text + " ", lp, ["a"]) == {}  # tokens don't spell the text
    odd = tokens(('{"a": {"answer": "', None), ("maybe", None), ('"}}', None))
    assert yes_probabilities('{"a": {"answer": "maybe"}}', odd, ["a"]) == {}


def test_state_mapping_and_images_in_messages() -> None:
    msgs = decision_messages({"b": 1, "a": "é"}, {"q": dict(YES_NO)}, ["x.png"])
    assert msgs[1]["content"][0]["text"] == json.dumps({"b": 1, "a": "é"}, indent=2, ensure_ascii=False)
    assert msgs[1]["content"][1] == {"type": "image", "path": "x.png"}


def test_emulated_images_go_through_capability_checks(isolated) -> None:
    (isolated / "a.png").write_bytes(b"png")
    with (
        respx.mock(assert_all_mocked=True),
        pytest.raises(mk.errors.CapabilityError, match="cannot read images"),
    ):
        mk.decision("gemma4-12b", sink=mk.records.NullSink()).decide("x", {"q": YES_NO}, images=["a.png"])


def test_decision_factory_errors(isolated) -> None:
    with pytest.raises(ConfigError, match="embedding model"):
        mk.decision("nomic-embed-text")
    path = isolated / "r.toml"
    path.write_text('[models.odd]\nprovider = "ollama"\nkind = "decision"\n')
    dm = mk.decision("odd", registry=mk.registry.load([path]), sink=mk.records.NullSink())
    with pytest.raises(ConfigError, match="no native decision provider 'ollama'"):
        dm.decide("x", {"q": YES_NO})


def test_jev_mapping_edge_cases() -> None:
    qs = {"c": {"type": "choice", "instructions": "x", "options": ["a", "b"]}, "y": dict(YES_NO)}
    data = {"answers": [{"id": "c", "probabilities": {"zzz": 0.9, "a": 0.1}}, {"id": "y", "p_yes": "n/a"}]}
    a = from_response(data, qs, calibrated=True)
    assert "not one of the options" in a["c"]["error"]
    assert a["y"]["value"] is None
    assert a["y"]["error"]


@pytest.mark.parametrize(
    ("item", "error"),
    [
        ({"id": "y", "p_yes": 1.5}, "outside [0, 1]"),
        ({"id": "y", "p_yes": -0.1}, "outside [0, 1]"),
        ({"id": "y", "p_yes": "nan"}, "outside [0, 1]"),
        ({"id": "c", "probabilities": {"a": 0.9, "b": 0.8}}, "sum above 1"),
        ({"id": "c", "probabilities": {"a": 1.2}}, "outside [0, 1]"),
    ],
)
def test_jev_out_of_range_probabilities_become_errors(item, error) -> None:
    qs = {"c": {"type": "choice", "instructions": "x", "options": ["a", "b"]}, "y": dict(YES_NO)}
    other = {"id": "y", "p_yes": 0.2}
    if item["id"] == "y":
        other = {"id": "c", "probabilities": {"a": 0.7, "b": 0.3}}
    a = from_response({"answers": [item, other]}, qs, calibrated=True)
    bad, good = a[item["id"]], a[other["id"]]
    assert bad["value"] is None
    assert error in bad["error"]
    assert good["error"] is None
    assert good["value"] in (0.7, 0.2)


def test_jev_non_object_answer_items_are_ignored() -> None:
    a = from_response({"answers": ["y", 3, {"id": "y", "p_yes": 0.25}]}, {"y": dict(YES_NO)}, calibrated=True)
    assert a["y"]["value"] == 0.25
    a = from_response({"answers": ["y"]}, {"y": dict(YES_NO)}, calibrated=True)
    assert a["y"]["value"] is None
    assert "missing from the response" in a["y"]["error"]


def test_jev_non_object_reply_and_custom_base_url(isolated, monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "tsk-0000000000")
    path = isolated / "r.toml"
    path.write_text('[models.jev]\nbase_url = "http://jev.local/api/"\n')
    dm = mk.decision("jev", registry=mk.registry.load([path]), sink=mk.records.NullSink())
    with respx.mock(base_url="http://jev.local/api") as mock:
        mock.post("/decide").respond(json=[1, 2])
        with pytest.raises(ProviderError, match="expected a JSON object"):
            dm.decide("x", {"q": YES_NO})
