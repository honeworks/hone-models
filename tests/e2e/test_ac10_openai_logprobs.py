"""AC-10: OpenAI-compatible provider incl. logprobs -> calibrated yes/no from the token logprobs."""

import json
import math
import re

import pytest
import respx

import hone_models as mk

pytestmark = pytest.mark.e2e
BASE = "http://localhost:8080/v1"
REGISTRY = f"""
[models.judge]
provider = "openai_compatible"
model = "local-judge"
base_url = "{BASE}"
[models.judge.capabilities]
json_schema = true
logprobs = {{logprobs}}
"""


def judge(isolated, logprobs: bool) -> mk.DecisionClient:
    path = isolated / "reg.toml"
    path.write_text(REGISTRY.replace("{logprobs}", str(logprobs).lower()))
    return mk.decision("judge", registry=mk.registry.load([path]), sink=mk.records.MemorySink())


def test_ac10_yes_no_calibrated_from_logprobs(isolated, recorded) -> None:
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/chat/completions").respond(json=recorded("openai_decide_logprobs.json"))
        a = judge(isolated, logprobs=True).decide("Customers see 500s.", {"urgent": mk.YesNo("Urgent?")})
    body = json.loads(route.calls.last.request.content)
    assert body["logprobs"] is True
    assert body["response_format"]["type"] == "json_schema"
    yes, no = math.exp(-0.105) + math.exp(-5.0), math.exp(-2.303)
    assert a["urgent"]["value"] == pytest.approx(yes / (yes + no))
    assert a["urgent"]["value"] == pytest.approx(0.90, abs=0.01)  # not the stated confidence (0.6)
    assert a["urgent"]["calibrated"] is True
    assert a["urgent"]["confidence"] == 0.6
    assert a["urgent"]["probabilities"]["no"] == pytest.approx(1 - a["urgent"]["value"])


def test_ac10_without_logprobs_uses_stated_confidence(isolated, recorded) -> None:
    reply = recorded("openai_decide_logprobs.json")
    del reply["choices"][0]["logprobs"]
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/chat/completions").respond(json=reply)
        a = judge(isolated, logprobs=False).decide("Customers see 500s.", {"urgent": mk.YesNo("Urgent?")})
    assert "logprobs" not in json.loads(route.calls.last.request.content)
    assert a["urgent"]["value"] == 0.6
    assert a["urgent"]["calibrated"] is False


def reply_with_logprobs(answers: dict, tops: list[dict[str, float]]) -> dict:
    """A chat reply whose token logprobs spell `answers` as JSON; the n-th yes/no token gets `tops[n]`."""
    content = json.dumps(answers)
    tokens, pending = [], list(tops)
    for tok in re.findall(r"\w+|\W+", content):
        top = pending.pop(0) if tok in ("yes", "no") else {tok: -0.001}
        tokens.append({"token": tok, "logprob": top.get(tok, -0.001), "top_logprobs": _top(top)})
    message = {"role": "assistant", "content": content}
    return {"model": "local-judge", "choices": [{"message": message, "logprobs": {"content": tokens}}]}


def _top(top: dict[str, float]) -> list[dict]:
    return [{"token": t, "logprob": lp} for t, lp in top.items()]


def test_ac10_logprobs_capable_but_missing_in_reply_falls_back(isolated, recorded) -> None:
    reply = recorded("openai_decide_logprobs.json")
    del reply["choices"][0]["logprobs"]
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/chat/completions").respond(json=reply)
        a = judge(isolated, logprobs=True).decide("Customers see 500s.", {"urgent": mk.YesNo("Urgent?")})
    assert json.loads(route.calls.last.request.content)["logprobs"] is True
    assert a["urgent"]["value"] == 0.6
    assert a["urgent"]["calibrated"] is False


def test_ac10_each_yes_no_reads_its_own_token_other_types_stay_uncalibrated(isolated) -> None:
    answers = {
        "urgent": {"answer": "yes", "confidence": 0.6, "rationale": "r"},
        "blame": {"answer": "no", "confidence": 0.7, "rationale": "r"},
        "team": {"answer": "infra", "confidence": 0.5, "rationale": "r"},
        "severity": {"answer": 4, "confidence": 0.9, "rationale": "r"},
    }
    tops = [{"yes": math.log(0.8), "no": math.log(0.2)}, {"no": math.log(0.3), "yes": math.log(0.7)}]
    questions = {
        "urgent": mk.YesNo("Urgent?"),
        "blame": mk.YesNo("Intern's fault?"),
        "team": mk.Choice(["infra", "backend"], "Team?"),
        "severity": mk.ScoreQ("Severity?"),
    }
    with respx.mock(base_url=BASE) as mock:
        mock.post("/chat/completions").respond(json=reply_with_logprobs(answers, tops))
        a = judge(isolated, logprobs=True).decide("Customers see 500s.", questions)
    assert a["urgent"]["value"] == pytest.approx(0.8)
    assert a["blame"]["value"] == pytest.approx(0.7)
    assert a["urgent"]["calibrated"] is a["blame"]["calibrated"] is True
    assert (a["team"]["value"], a["team"]["calibrated"]) == (0.5, False)
    assert (a["severity"]["raw"], a["severity"]["calibrated"]) == (4, False)
