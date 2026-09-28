"""AC-3: fenced JSON -> parsed; truncated JSON -> repaired; invalid then valid -> retried."""

import json

import pytest
import respx
from pydantic import BaseModel

import hone_models as mk

pytestmark = pytest.mark.e2e
URL = "http://127.0.0.1:11434"


class Verdict(BaseModel):
    score: int
    rationale: str


class Ideas(BaseModel):
    ideas: list[str]


def reply(content: str, done_reason: str = "stop") -> dict:
    return {"model": "m", "message": {"content": content}, "done_reason": done_reason, "eval_count": 5}


def adhoc_llm(mock) -> mk.TextClient:
    """An ad-hoc Ollama model (probed; Ollama always supports `format`)."""
    mock.post("/api/show").respond(json={"capabilities": ["completion"], "model_info": {}})
    return mk.text("ollama:plain:1b", sink=mk.records.MemorySink())


def test_ac3_fenced_json_is_parsed() -> None:
    with respx.mock(base_url=URL) as mock:
        llm = adhoc_llm(mock)
        chat = mock.post("/api/chat").respond(
            json=reply('Sure!\n```json\n{"score": 4, "rationale": "catchy"}\n```\nHope this helps!')
        )
        r = llm.complete([{"role": "user", "content": "Rate this pitch 1-5 ..."}], schema=Verdict)
    sent = json.loads(chat.calls[0].request.content)
    assert sent["messages"][0]["role"] == "system"
    assert json.dumps(Verdict.model_json_schema()) in sent["messages"][0]["content"]
    assert sent["format"] == Verdict.model_json_schema()
    assert llm.sink.spans[0]["attributes"]["hone.models.structured.schema"] == Verdict.model_json_schema()  # type: ignore[attr-defined]
    assert r.parsed == Verdict(score=4, rationale="catchy")
    assert r.structured_path == "parsed"
    assert r.attempts == 1
    assert r.error is None


def test_ac3_truncated_json_is_repaired() -> None:
    truncated = '{"ideas": ["Cargo Hold Chaos", "Mermaid Mutiny", "Anchors Aw'
    with respx.mock(base_url=URL) as mock:
        llm = adhoc_llm(mock)
        mock.post("/api/chat").respond(json=reply(truncated, "length"))
        r = llm.complete([{"role": "user", "content": "12 ideas"}], schema=Ideas)
    assert r.parsed == Ideas(ideas=["Cargo Hold Chaos", "Mermaid Mutiny"])
    assert r.structured_path == "repaired"
    assert r.finish_reason == "length"  # always reported
    span = llm.sink.spans[0]  # type: ignore[attr-defined]
    assert span["attributes"]["hone.models.structured.path"] == "repaired"
    assert span["attributes"]["gen_ai.response.finish_reasons"] == ["length"]


def test_ac3_invalid_then_valid_is_retried_with_error_sent_back() -> None:
    with respx.mock(base_url=URL) as mock:
        llm = adhoc_llm(mock)
        chat = mock.post("/api/chat")
        chat.side_effect = [
            respx.MockResponse(json=reply('{"score": "high"}')),
            respx.MockResponse(json=reply('{"score": 5, "rationale": "great hook"}')),
        ]
        r = llm.complete([{"role": "user", "content": "Rate it"}], schema=Verdict)
    assert r.parsed == Verdict(score=5, rationale="great hook")
    assert r.structured_path == "retried"
    assert r.attempts == 2
    assert r.usage["output_tokens"] == 10  # both attempts counted
    second = json.loads(chat.calls[1].request.content)["messages"]
    assert second[-2] == {"role": "assistant", "content": '{"score": "high"}'}
    assert second[-1]["role"] == "user"
    assert "not valid" in second[-1]["content"]
    assert "rationale" in second[-1]["content"]  # the validation error names the missing field
    span = llm.sink.spans[0]  # type: ignore[attr-defined]
    assert span["attributes"]["hone.models.structured.attempts"] == 2
    assert [e["name"] for e in span["events"]] == ["structured_retry"]


def test_ac3_constrained_when_supported_and_failed_when_hopeless() -> None:
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat").respond(json=reply('{"ok": true}'))
        r = mk.text("gemma4-12b", sink=mk.records.NullSink()).complete(
            [{"role": "user", "content": "ok?"}], schema=schema
        )
        sent = json.loads(chat.calls.last.request.content)
        assert sent["format"] == schema  # top level, not inside options
        assert "format" not in sent["options"]
        assert r.parsed == {"ok": True}
        assert r.structured_path == "constrained"

        mock.post("/api/chat").respond(json=reply("I cannot answer that."))
        r = mk.text("gemma4-12b", sink=mk.records.NullSink()).complete(
            [{"role": "user", "content": "ok?"}], schema=schema
        )
    assert r.parsed is None
    assert r.structured_path == "failed"
    assert r.attempts == 3
    assert r.error is not None
