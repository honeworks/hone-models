"""AC-1: mk.text("ollama:<model>") chat against a mock Ollama -> text, usage, finish_reason, gen_ai.* span."""

import json

import httpx
import pytest
import respx

import hone_models as mk

pytestmark = pytest.mark.e2e
URL = "http://127.0.0.1:11434"


def test_ac1_ollama_chat(recorded, isolated) -> None:
    sink = mk.records.SqliteSpanSink(isolated / "spans.db")
    with respx.mock(base_url=URL, assert_all_called=True) as mock:
        mock.post("/api/show").respond(json=recorded("ollama_show_llama.json"))
        chat = mock.post("/api/chat").respond(json=recorded("ollama_chat_ok.json"))
        llm = mk.text("ollama:llama3.2:1b", sink=sink)
        r = llm.complete([{"role": "user", "content": "Write a haiku about rain."}], temperature=0.2, seed=7)

    assert r.text.startswith("Rain taps the window")
    assert r.usage == {"input_tokens": 31, "output_tokens": 22}
    assert r.finish_reason == "stop"
    assert r.error is None
    assert r.model == "llama3.2:1b"
    assert r.parsed is None
    assert r.structured_path is None

    sent = json.loads(chat.calls.last.request.content)
    assert sent["model"] == "llama3.2:1b"
    assert sent["stream"] is False
    assert sent["options"]["temperature"] == 0.2
    assert sent["options"]["seed"] == 7
    assert sent["options"]["num_ctx"] == 4096  # (8 prompt + 2048 output) * 1.1, rounded up to 2048 steps
    assert sent["options"]["num_predict"] == 2048  # always bounded, so truncation shows as "length"
    assert "format" not in sent
    assert "think" not in sent  # probe: not a thinking model

    [span] = mk.records.read_spans(isolated / "spans.db")
    assert span["span_id"] == r.span_id
    assert span["name"] == "hone.models.chat"
    assert span["kind"] == "client"
    assert span["status"]["code"] == "ok"
    a = span["attributes"]
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.provider.name"] == "ollama"
    assert a["gen_ai.request.model"] == "llama3.2:1b"
    assert a["gen_ai.response.model"] == "llama3.2:1b"
    assert a["gen_ai.request.temperature"] == 0.2
    assert a["gen_ai.request.seed"] == 7
    assert a["gen_ai.response.finish_reasons"] == ["stop"]
    assert a["gen_ai.usage.input_tokens"] == 31
    assert a["gen_ai.usage.output_tokens"] == 22
    assert a["gen_ai.input.messages"] == [{"role": "user", "content": "Write a haiku about rain."}]
    assert a["gen_ai.output.messages"][0]["content"] == r.text
    assert a["hone.models.model_id"] == "ollama:llama3.2:1b"
    assert a["hone.models.context.limit"] == 131072  # probed via /api/show
    assert a["hone.schema_version"] == "1"
    assert a["gen_ai.request.max_tokens"] == 2048
    assert a["hone.models.context.estimated_prompt_tokens"] == 8
    assert len(span["trace_id"]) == 32
    assert span["parent_span_id"] is None


def test_ac1_same_call_same_request_and_record(recorded) -> None:
    sink = mk.records.MemorySink()
    bodies = []
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat").respond(json=recorded("ollama_chat_ok.json"))
        llm = mk.text("gemma4-12b", sink=sink)
        for _ in range(2):
            llm.complete([{"role": "user", "content": "Write a haiku."}], seed=7)
            bodies.append(chat.calls.last.request.content)
    assert bodies[0] == bodies[1]
    first, second = (s["attributes"] for s in sink.spans)
    assert first == second


def test_ac1_unknown_token_counts_are_not_zero() -> None:
    reply = {"model": "m", "message": {"content": "cached"}, "done_reason": "stop", "eval_count": 4}
    sink = mk.records.MemorySink()
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(json=reply)
        r = mk.text("gemma4-12b", sink=sink).complete([{"role": "user", "content": "x"}])
    assert r.usage == {"output_tokens": 4}
    assert "gen_ai.usage.input_tokens" not in sink.spans[0]["attributes"]


def test_ac1_server_down_raises_provider_error(no_backoff) -> None:
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").mock(side_effect=httpx.ConnectError("refused"))
        llm = mk.text("gemma4-12b")
        with pytest.raises(mk.errors.ProviderError, match="failed after 3 attempts"):
            llm.complete([{"role": "user", "content": "hi"}])
