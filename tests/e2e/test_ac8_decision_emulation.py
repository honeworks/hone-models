"""AC-8: decision emulation on an LLM with a fake transport -> answer normalization (design §6), per-question
error handling, contract checker passes."""

import json

import pytest
import respx

import hone_models as mk
from select_contracts import check_decision_client

pytestmark = pytest.mark.e2e
URL = "http://127.0.0.1:11434"
QUESTIONS = {
    "urgent": mk.YesNo("Does this need attention now?"),
    "team": mk.Choice(["infra", "backend", "frontend"], "Which team owns it?"),
    "severity": mk.ScoreQ("How severe?", scale=(1, 5), anchors={1: "cosmetic", 5: "outage"}),
}
STATE = "The deploy failed twice and customers see 500s."


def ollama_reply(answers: dict) -> dict:
    content = json.dumps(answers)
    return {"model": "gemma4-12b:latest", "message": {"content": content}, "done_reason": "stop"}


GOOD = {
    "urgent": {"answer": "yes", "confidence": 0.8, "rationale": "Customers are affected."},
    "team": {"answer": "infra", "confidence": 0.7, "rationale": "Deploys are infra."},
    "severity": {"answer": 4, "confidence": 0.9, "rationale": "Errors for users."},
}


def test_ac8_emulated_decision_normalized() -> None:
    sink = mk.records.MemorySink()
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat").respond(json=ollama_reply(GOOD))
        a = mk.decision("gemma4-12b", sink=sink).decide(STATE, QUESTIONS)

    assert chat.call_count == 1  # one call answers every question
    body = json.loads(chat.calls.last.request.content)
    assert body["options"]["temperature"] == 0.0
    schema = body["format"]
    assert schema["required"] == ["urgent", "team", "severity"]
    assert schema["properties"]["team"]["properties"]["answer"]["enum"] == ["infra", "backend", "frontend"]
    system = body["messages"][0]["content"]
    assert "Which team owns it?" in system
    assert "1 = cosmetic" in system
    assert body["messages"][1] == {"role": "user", "content": STATE}

    assert a["urgent"] == {
        "type": "yes_no",
        "value": 0.8,
        "choice": None,
        "probabilities": {"yes": 0.8, "no": pytest.approx(0.2)},
        "raw": None,
        "confidence": 0.8,
        "calibrated": False,
        "rationale": "Customers are affected.",
        "error": None,
    }
    assert a["team"]["choice"] == "infra"
    assert a["team"]["value"] == 0.7
    assert a["team"]["probabilities"] == pytest.approx({"infra": 0.7, "backend": 0.15, "frontend": 0.15})
    assert (a["severity"]["raw"], a["severity"]["value"]) == (4.0, 0.75)
    assert a["severity"]["calibrated"] is False

    chat_span, decide = sorted(sink.spans, key=lambda s: s["name"])
    assert decide["name"] == "hone.models.decide"
    assert chat_span["name"] == "hone.models.chat"
    assert chat_span["parent_span_id"] == decide["span_id"]
    assert chat_span["trace_id"] == decide["trace_id"]
    attrs = decide["attributes"]
    assert attrs["hone.models.decision.questions"]["severity"]["scale"] == [1, 5]
    assert attrs["hone.models.decision.answers"] == a
    assert attrs["hone.models.decision.state"] == STATE
    assert attrs["hone.models.model_id"] == "gemma4-12b"
    assert decide["status"]["code"] == "ok"


def test_ac8_per_question_errors(no_backoff) -> None:
    bad = {**GOOD, "team": {"answer": "database", "confidence": 0.9, "rationale": "?"}}
    bad["severity"] = {"answer": 9, "confidence": 0.5, "rationale": "?"}
    sink = mk.records.MemorySink()
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat").respond(json=ollama_reply(bad))
        a = mk.decision("gemma4-12b", sink=sink).decide(STATE, QUESTIONS)
    assert chat.call_count == 3  # validation errors were sent back twice before giving up
    assert a["urgent"]["value"] == 0.8
    assert a["urgent"]["error"] is None
    for name in ("team", "severity"):
        assert a[name]["value"] is None
        assert a[name]["error"]
    assert "database" in a["team"]["error"]
    assert "outside the scale" in a["severity"]["error"]
    assert next(s for s in sink.spans if s["name"] == "hone.models.decide")["status"]["code"] == "ok"


def test_ac8_unusable_reply_fails_every_question() -> None:
    sink = mk.records.MemorySink()
    empty = {"model": "m", "message": {"content": ""}, "done_reason": "stop"}
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(json=empty)
        a = mk.decision("gemma4-12b", sink=sink).decide(STATE, QUESTIONS)
    assert all(ans["value"] is None and ans["error"] for ans in a.values())
    decide = next(s for s in sink.spans if s["name"] == "hone.models.decide")
    assert decide["status"]["code"] == "error"


def test_ac8_contract_checker_passes() -> None:
    reply = {
        "q1": {"answer": "yes", "confidence": 0.95, "rationale": "It says blue."},
        "q2": {"answer": "blue", "confidence": 0.9, "rationale": "Blue."},
        "q3": {"answer": 3, "confidence": 0.6, "rationale": "Plain."},
    }
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(json=ollama_reply(reply))
        check_decision_client(mk.decision("gemma4-12b", sink=mk.records.NullSink()))


def test_ac8_transport_failure_raises_and_both_spans_fail() -> None:
    sink = mk.records.MemorySink()
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(400, json={"error": "bad request"})
        with pytest.raises(mk.errors.ProviderError, match="HTTP 400"):
            mk.decision("gemma4-12b", sink=sink).decide(STATE, QUESTIONS)
    assert {s["name"]: s["status"]["code"] for s in sink.spans} == {
        "hone.models.chat": "error",
        "hone.models.decide": "error",
    }


def test_ac8_trace_is_inherited_and_invented_answers_dropped() -> None:
    sink = mk.records.MemorySink()
    trace = {"traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"}
    reply = {**GOOD, "invented": {"answer": "yes", "confidence": 1.0, "rationale": "?"}}
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(json=ollama_reply(reply))
        a = mk.decision("gemma4-12b", sink=sink).decide(STATE, QUESTIONS, trace=trace)
    assert set(a) == set(QUESTIONS)
    assert sum(a["team"]["probabilities"].values()) == pytest.approx(1.0)
    decide = next(s for s in sink.spans if s["name"] == "hone.models.decide")
    assert (decide["trace_id"], decide["parent_span_id"]) == ("a" * 32, "b" * 16)
