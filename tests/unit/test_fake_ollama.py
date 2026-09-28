"""The public FakeOllama: default replies, scripted answers (`queue`), the OpenAI-compatible endpoints."""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import hone_models as mk
from hone_models.errors import ProviderError
from hone_models.registry import Registry
from hone_models.testing import FakeOllama
from hone_models.testing.fake_ollama import reply, sample

USER = [{"role": "user", "content": "hi"}]


def openai_registry(url: str) -> Registry:
    """A registry with one OpenAI-compatible chat and one embedding model served by the fake's `/v1`."""
    Path("hone-models.toml").write_text(
        f"""
[models.chat]
provider = "openai_compatible"
model = "fake-chat"
base_url = "{url}/v1"
capabilities = {{ json_schema = true }}

[models.emb]
provider = "openai_compatible"
model = "fake-embed"
kind = "embedding"
base_url = "{url}/v1"
"""
    )
    return mk.registry.load()


def test_queued_dict_is_merged_over_the_default_reply_then_defaults_resume() -> None:
    llm = mk.text("ollama:m", sink=mk.records.NullSink())
    with FakeOllama() as server:
        server.queue(
            "/api/chat", {"message": {"role": "assistant", "content": "scripted"}, "done_reason": "length"}
        )
        first, second = llm.complete(USER), llm.complete(USER)
    assert (first.text, first.finish_reason) == ("scripted", "length")
    assert first.usage["input_tokens"] == 0  # kept from the default reply
    assert (second.text, second.finish_reason) == ("echo: hi", "stop")


def test_queued_status_is_an_http_error(no_backoff) -> None:
    llm = mk.text("ollama:m", sink=mk.records.NullSink())
    with FakeOllama() as server:
        server.queue("/api/chat", 400)
        with pytest.raises(ProviderError) as info:
            llm.complete(USER)
        assert info.value.status == 400
        server.queue("/api/chat", 503)  # transient: retried, then the default reply
        assert llm.complete(USER).text == "echo: hi"
    assert [r["path"] for r in server.requests].count("/api/chat") == 3


def test_queues_are_per_path_and_keep_their_order(no_backoff) -> None:
    sink = mk.records.NullSink()
    with FakeOllama() as server:
        server.queue("/api/embed", 400)
        server.queue("/api/chat", 404, {"message": {"role": "assistant", "content": "second"}})
        with pytest.raises(ProviderError) as chat_error:
            mk.text("ollama:m", sink=sink).complete(USER)
        assert mk.text("ollama:m", sink=sink).complete(USER).text == "second"
        assert mk.text("ollama:m", sink=sink).complete(USER).text == "echo: hi"
        with pytest.raises(ProviderError) as embed_error:  # still queued: chats did not consume it
            mk.embedder("ollama:e", sink=sink).embed(["x"])
    assert (chat_error.value.status, embed_error.value.status) == (404, 400)


def test_requests_record_path_without_query_body_and_headers() -> None:
    with FakeOllama() as server:
        server.queue("/api/version", {"version": "9.9"})
        with urllib.request.urlopen(f"{server.url}/api/version?verbose=1") as response:  # noqa: S310
            assert json.load(response) == {"version": "9.9", "done": True}
    assert server.requests[-1]["path"] == "/api/version"
    assert "Host" in server.requests[-1]["headers"]


def test_sample_follows_refs_any_of_and_arrays() -> None:
    schema = {
        "$defs": {"A": {"enum": ["x", "y"]}},
        "type": "object",
        "properties": {
            "a": {"$ref": "#/$defs/A"},
            "b": {"anyOf": [{"type": "boolean"}, {"type": "string"}]},
            "c": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 2}},
        },
    }
    assert sample(schema) == {"a": "x", "b": True, "c": [1.5]}
    assert reply("/api/version", {}) == {"version": "0.0.0-fake", "done": True}


def test_openai_compatible_chat_and_embeddings() -> None:
    sink = mk.records.NullSink()
    schema = {"type": "object", "properties": {"n": {"type": "integer", "maximum": 4}}, "required": ["n"]}
    with FakeOllama() as server:
        reg = openai_registry(server.url)
        llm = mk.text("chat", registry=reg, sink=sink)
        text = llm.complete([{"role": "user", "content": [{"type": "text", "text": "parts"}]}])
        structured = llm.complete(USER, schema=schema)
        vectors = mk.embedder("emb", registry=reg, sink=sink).embed(["a", "b"])
    assert (text.text, text.usage["output_tokens"]) == ("echo: parts", 3)
    assert (structured.parsed, structured.structured_path) == ({"n": 4}, "constrained")
    assert len(vectors) == 2
    assert vectors[0] != vectors[1]
    assert server.requests[1]["body"]["response_format"]["json_schema"]["schema"] == schema


def test_openai_compatible_error_status(no_backoff) -> None:
    with FakeOllama() as server:
        server.queue("/v1/chat/completions", 400)
        llm = mk.text("chat", registry=openai_registry(server.url), sink=mk.records.NullSink())
        with pytest.raises(ProviderError) as info:
            llm.complete(USER)
    assert info.value.status == 400


@pytest.mark.parametrize("before", ["http://elsewhere:1", None])
def test_start_and_stop_set_and_restore_ollama_host(before, monkeypatch: pytest.MonkeyPatch) -> None:
    if before:
        monkeypatch.setenv("OLLAMA_HOST", before)
    server = FakeOllama().start()
    assert os.environ["OLLAMA_HOST"] == server.url
    server.stop()
    assert os.environ.get("OLLAMA_HOST") == before
    with pytest.raises(urllib.error.URLError):  # really stopped
        urllib.request.urlopen(f"{server.url}/api/version", timeout=2)  # noqa: S310


def test_responder_answers_from_the_request_and_none_keeps_the_default() -> None:
    """Change 0007: answer by schema title; queued answers still come first."""
    from pydantic import BaseModel  # noqa: PLC0415

    class Lessons(BaseModel):
        titles: list[str]

    def responder(path: str, body: dict) -> dict | None:
        schema = body.get("format") or {}
        if path == "/api/chat" and schema.get("title") == "Lessons":
            content = json.dumps({"titles": ["One", "Two", "Three"]})
            return {"message": {"role": "assistant", "content": content}}
        return None

    with FakeOllama(responder=responder) as server:
        llm = mk.text("gemma4-12b", sink=mk.records.NullSink())
        assert llm.complete([{"role": "user", "content": "plan"}], schema=Lessons).parsed == Lessons(
            titles=["One", "Two", "Three"]
        )
        assert llm.complete([{"role": "user", "content": "hi"}]).text == "echo: hi"
        server.queue("/api/chat", {"message": {"role": "assistant", "content": '{"titles": []}'}})
        assert llm.complete([{"role": "user", "content": "plan"}], schema=Lessons).parsed == Lessons(
            titles=[]
        )
