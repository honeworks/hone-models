"""AC-2: a thinking model returns empty content + filled thinking -> think=false sent; error, not ''."""

import json

import pytest
import respx

import hone_models as mk

pytestmark = pytest.mark.e2e
URL = "http://127.0.0.1:11434"


def test_ac2_thinking_model(recorded) -> None:
    sink = mk.records.MemorySink()
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat").respond(json=recorded("ollama_chat_thinking_only.json"))
        r = mk.text("gemma4-12b", sink=sink).complete([{"role": "user", "content": "A haiku, please."}])

    assert json.loads(chat.calls.last.request.content)["think"] is False  # registry: thinking = true
    assert r.text == ""
    assert r.error is not None
    assert "thinking" in r.error
    assert sink.spans[0]["status"] == {"code": "error", "message": r.error}
    assert r.finish_reason == "length"
    assert r.usage["output_tokens"] == 2048


def test_ac2_probed_thinking_model_and_schema(recorded) -> None:
    show = {"capabilities": ["completion", "thinking"], "model_info": {"qwen3.context_length": 40960}}
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/show").respond(json=show)
        chat = mock.post("/api/chat").respond(json=recorded("ollama_chat_thinking_only.json"))
        r = mk.text("ollama:qwen3:4b", sink=mk.records.NullSink()).complete(
            [{"role": "user", "content": "Return {}"}], schema={"type": "object"}
        )
    assert json.loads(chat.calls.last.request.content)["think"] is False
    assert r.parsed is None
    assert r.error is not None
    assert "thinking" in r.error


def test_ac2_caller_can_ask_for_reasoning(recorded) -> None:
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat").respond(json=recorded("ollama_chat_ok.json"))
        r = mk.text("gemma4-12b", sink=mk.records.NullSink()).complete(
            [{"role": "user", "content": "Think it through."}], think=True
        )
    assert json.loads(chat.calls.last.request.content)["think"] is True
    assert r.error is None


def test_ac2_plain_empty_reply_is_an_error() -> None:
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(
            json={"model": "m", "message": {"content": "  "}, "done_reason": "stop"}
        )
        r = mk.text("gemma4-12b", sink=mk.records.NullSink()).complete([{"role": "user", "content": "x"}])
    assert r.error == "model returned an empty response"
