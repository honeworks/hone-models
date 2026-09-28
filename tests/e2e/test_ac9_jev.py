"""AC-9: Jev provider with recorded fixtures -> mapping to Answers, calibrated=True, contract checker passes.

The Jev API shape is unverified (design/decisions.md D-019); see src/hone_models/providers/jev.py."""

import json

import pytest
import respx

import hone_models as mk
from select_contracts import check_decision_client

pytestmark = pytest.mark.e2e
JEV = "https://api.typesafe.ai/v1"
KEY = "tsk-planted-secret-000000"
QUESTIONS = {
    "urgent": mk.YesNo("Does this need attention now?"),
    "team": mk.Choice(["infra", "backend", "frontend"], "Which team owns it?"),
    "severity": mk.ScoreQ("How severe?", scale=(1, 5)),
    "blame": mk.YesNo("Is it the intern's fault?"),
}


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    return KEY


def test_ac9_jev_answers_are_mapped_and_calibrated(key, recorded, isolated) -> None:
    (isolated / "frame.png").write_bytes(b"\x89PNG")
    sink = mk.records.SqliteSpanSink(isolated / "spans.db")
    with respx.mock(base_url=JEV) as mock:
        route = mock.post("/decide").respond(json=recorded("jev_decide_ok.json"))
        a = mk.decision("jev", sink=sink).decide(
            "The deploy failed twice.", QUESTIONS, images=[str(isolated / "frame.png")]
        )
    request = route.calls.last.request
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body["state"] == "The deploy failed twice."
    assert body["questions"][1] == {
        "id": "team",
        "type": "choice",
        "text": "Which team owns it?",
        "options": ["infra", "backend", "frontend"],
    }
    assert body["images"] == [{"data": "iVBORw==", "mime": "image/png"}]

    assert a["urgent"]["value"] == 0.91
    assert a["urgent"]["probabilities"] == {"yes": 0.91, "no": pytest.approx(0.09)}
    assert a["team"]["choice"] == "infra"
    assert a["team"]["value"] == 0.72
    assert a["severity"]["raw"] == 4.2
    assert a["severity"]["value"] == pytest.approx(0.8)
    for name in ("urgent", "team", "severity"):
        assert a[name]["calibrated"] is True
        assert a[name]["error"] is None
    assert a["blame"]["value"] is None
    assert "outside the model's domain" in a["blame"]["error"]

    sink.flush()
    raw = (isolated / "spans.db").read_bytes()
    assert KEY.encode() not in raw
    span = mk.records.read_spans(isolated / "spans.db")[0]
    assert span["name"] == "hone.models.decide"
    attrs = span["attributes"]
    assert attrs["gen_ai.provider.name"] == "jev"
    assert attrs["gen_ai.response.model"] == "jev-1-2026-09"
    assert attrs["gen_ai.usage.input_tokens"] == 57
    assert attrs["hone.models.cost_usd"] == pytest.approx(57 * 0.042 / 1e6)


def test_ac9_missing_answers_and_missing_key(key, monkeypatch) -> None:
    with respx.mock(base_url=JEV) as mock:
        mock.post("/decide").respond(json={"answers": []})
        a = mk.decision("jev", sink=mk.records.NullSink()).decide("x", {"ok": mk.YesNo("Ok?")})
    assert a["ok"]["value"] is None
    assert "missing from the response" in a["ok"]["error"]

    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(mk.errors.ConfigError, match="TYPESAFE_API_KEY"):
        mk.decision("jev", sink=mk.records.NullSink()).decide("x", {"ok": mk.YesNo("Ok?")})


def test_ac9_cost_is_recorded_without_output_tokens(key) -> None:
    """The response shape in providers/jev.py reports input tokens only; Jev prices only input."""
    sink = mk.records.MemorySink()
    reply = {"answers": [{"id": "ok", "p_yes": 0.8}], "usage": {"input_tokens": 57}}
    with respx.mock(base_url=JEV) as mock:
        mock.post("/decide").respond(json=reply)
        mk.decision("jev", sink=sink).decide("x", {"ok": mk.YesNo("Ok?")})
    assert sink.spans[-1]["attributes"]["hone.models.cost_usd"] == pytest.approx(57 * 0.042 / 1e6)


def test_ac9_contract_checker_passes(key) -> None:
    reply = {
        "answers": [
            {"id": "q1", "p_yes": 0.97},
            {"id": "q2", "probabilities": {"blue": 0.9, "red": 0.1}},
            {"id": "q3", "score": 3.1},
        ]
    }
    with respx.mock(base_url=JEV) as mock:
        mock.post("/decide").respond(json=reply)
        check_decision_client(mk.decision("jev", sink=mk.records.NullSink()))


def test_ac9_registry_can_mark_jev_uncalibrated_and_anchors_are_sent(key, isolated) -> None:
    path = isolated / "reg.toml"
    path.write_text("[models.jev.capabilities]\ncalibrated = false\n")
    question = mk.ScoreQ("How severe?", anchors={1: "cosmetic", 5: "outage"})
    with respx.mock(base_url=JEV) as mock:
        route = mock.post("/decide").respond(json={"answers": [{"id": "s", "score": 2}]})
        dm = mk.decision("jev", registry=mk.registry.load([path]), sink=mk.records.NullSink())
        a = dm.decide("x", {"s": question})
    assert json.loads(route.calls.last.request.content)["questions"][0]["anchors"] == {
        "1": "cosmetic",
        "5": "outage",
    }
    assert (a["s"]["value"], a["s"]["calibrated"]) == (0.25, False)
