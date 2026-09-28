"""AC-15: replay with a section removed -> new span with `replay_of`; the request lacks the section;
other params identical."""

import hashlib
import json

import pytest
import respx

import hone_models as mk

pytestmark = pytest.mark.e2e
URL = "http://127.0.0.1:11434"
REPLY = {"model": "gemma4-12b:latest", "message": {"content": '{"title": "x"}'}, "done_reason": "stop"}
SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}


def song_prompt() -> mk.Prompt:
    return mk.Prompt(
        "song_ideas",
        "3",
        sections={
            "system": "You write song pitches.",
            "rules": "Exactly 12 ideas, JSONL.",
            "format_example": mk.Section('{"title": "Cargo Hold Chaos", ...}', version="7"),
            "brief": "Theme: {theme}",
        },
        variables={"theme": "sea shanties"},
    )


def test_ac15_replay_without_a_section(isolated) -> None:
    db = isolated / "spans.db"
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat").respond(json=REPLY)
        llm = mk.text("gemma4-12b", sink=mk.records.SqliteSpanSink(db))
        llm.complete(song_prompt(), schema=SCHEMA, temperature=0.3, seed=7, max_tokens=300)
        original = mk.records.read_spans(db)[0]

        new = mk.replay.Replayer(sink=mk.records.SqliteSpanSink(db)).replay_call(
            original, {"prompt.sections": {"format_example": None}}
        )
        first, second = (json.loads(c.request.content) for c in chat.calls)

    assert new["attributes"]["hone.models.replay_of"] == original["span_id"]
    assert new["span_id"] != original["span_id"]
    stored = {s["span_id"]: s for s in mk.records.read_spans(db)}
    assert stored[new["span_id"]]["attributes"]["hone.models.replay_of"] == original["span_id"]

    assert "Cargo Hold Chaos" in first["messages"][1]["content"]
    assert "Cargo Hold Chaos" not in second["messages"][1]["content"]
    assert second["messages"][1]["content"] == "Exactly 12 ideas, JSONL.\n\nTheme: sea shanties"
    assert second["messages"][0] == first["messages"][0]
    assert {k: v for k, v in second.items() if k != "messages"} == {
        k: v for k, v in first.items() if k != "messages"
    }
    assert second["options"]["temperature"] == 0.3
    assert second["options"]["seed"] == 7
    assert second["format"] == SCHEMA

    ids = [s["id"] for s in new["attributes"]["hone.models.prompt.sections"]]
    assert ids == ["system", "rules", "brief"]
    assert new["attributes"]["hone.models.prompt.template_id"] == "song_ideas"


def test_ac15_overrides_model_params_and_messages() -> None:
    sink = mk.records.MemorySink()
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat").respond(json=REPLY)
        mk.text("gemma4-12b", sink=sink).complete([{"role": "user", "content": "hi"}], temperature=0.3)
        replayer = mk.replay.Replayer(sink=sink)
        new = replayer.replay_call(sink.spans[0], {"model": "qwen2.5vl-7b", "params": {"seed": 1}})
        body = json.loads(chat.calls.last.request.content)
        assert body["model"] == "qwen2.5vl:7b"
        assert (body["options"]["temperature"], body["options"]["seed"]) == (0.3, 1)
        assert new["attributes"]["hone.models.model_id"] == "qwen2.5vl-7b"

        again = replayer.replay_call(sink.spans[0], {"messages": [{"role": "user", "content": "bye"}]})
        assert json.loads(chat.calls.last.request.content)["messages"] == [{"role": "user", "content": "bye"}]
    assert len(sink.spans) == 3
    assert new["attributes"]["hone.models.replay_of"] == again["attributes"]["hone.models.replay_of"]
    assert again["attributes"]["hone.models.replay_of"] == sink.spans[0]["span_id"]
    both = {"messages": [], "prompt.sections": {}}
    with pytest.raises(mk.errors.ConfigError, match="not both"):
        replayer.replay_call(sink.spans[0], both)


def test_ac15_replay_needs_captured_content() -> None:
    sink = mk.records.MemorySink(capture_content=False)
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(json=REPLY)
        mk.text("gemma4-12b", sink=sink).complete([{"role": "user", "content": "hi"}])
    replayer = mk.replay.Replayer(sink=mk.records.NullSink())
    with pytest.raises(mk.errors.ConfigError, match="content capture"):
        replayer.replay_call(sink.spans[0], {})
    with pytest.raises(mk.errors.ConfigError, match="unknown replay overrides"):
        replayer.replay_call(sink.spans[0], {"temperature": 1})


def test_ac15_section_overrides_are_checked() -> None:
    sink = mk.records.MemorySink()
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(json=REPLY)
        mk.text("gemma4-12b", sink=sink).complete([{"role": "user", "content": "hi"}])
        mk.text("gemma4-12b", sink=sink).complete(song_prompt())
    replayer = mk.replay.Replayer(sink=mk.records.NullSink())
    with pytest.raises(mk.errors.ConfigError, match="not made from an mk"):
        replayer.replay_call(sink.spans[0], {"prompt.sections": {"rules": None}})
    with pytest.raises(mk.errors.ConfigError, match=r"unknown sections \['nope'\]"):
        replayer.replay_call(sink.spans[1], {"prompt.sections": {"nope": None}})


def test_ac15_replace_a_section_text() -> None:
    sink = mk.records.MemorySink()
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat").respond(json=REPLY)
        mk.text("gemma4-12b", sink=sink).complete(song_prompt())
        new = mk.replay.Replayer(sink=sink).replay_call(
            sink.spans[0], {"prompt.sections": {"rules": "3 ideas."}}
        )
    user = json.loads(chat.calls.last.request.content)["messages"][1]["content"]
    assert user == '3 ideas.\n\n{"title": "Cargo Hold Chaos", ...}\n\nTheme: sea shanties'
    sections = {s["id"]: s for s in new["attributes"]["hone.models.prompt.sections"]}
    assert sections["rules"]["sha256"] == hashlib.sha256(b"3 ideas.").hexdigest()
    assert sections["format_example"]["version"] == "7"
    assert sections["rules"]["version"] == "3"


def test_ac15_failed_replay_is_recorded_in_the_callers_trace(no_backoff) -> None:
    sink = mk.records.MemorySink()
    trace = {"traceparent": "00-" + "c" * 32 + "-" + "d" * 16 + "-01"}
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat")
        chat.side_effect = [respx.MockResponse(json=REPLY), respx.MockResponse(400, json={"error": "bad"})]
        mk.text("gemma4-12b", sink=sink).complete([{"role": "user", "content": "hi"}])
        with pytest.raises(mk.errors.ProviderError):
            mk.replay.Replayer(sink=sink).replay_call(sink.spans[0], {}, trace=trace)
    failed = sink.spans[1]
    assert failed["attributes"]["hone.models.replay_of"] == sink.spans[0]["span_id"]
    assert failed["status"]["code"] == "error"
    assert (failed["trace_id"], failed["parent_span_id"]) == ("c" * 32, "d" * 16)


def test_ac15_capture_off_sink_gets_and_returns_hashed_content() -> None:
    source = mk.records.MemorySink()
    target = mk.records.MemorySink(capture_content=False)
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(json=REPLY)
        mk.text("gemma4-12b", sink=source).complete([{"role": "user", "content": "secret plan"}])
        new = mk.replay.Replayer(sink=target).replay_call(source.spans[0], {})
    assert "secret plan" not in json.dumps(new)
    assert "secret plan" not in json.dumps(target.spans)
    assert target.spans[0]["attributes"]["hone.models.replay_of"] == source.spans[0]["span_id"]


def test_ac15_span_from_another_recorder() -> None:
    span = {
        "span_id": "f" * 16,
        "attributes": {
            "gen_ai.provider.name": "ollama",
            "gen_ai.request.model": "llama3.2:1b",
            "gen_ai.request.temperature": 0.1,
            "gen_ai.input.messages": [{"role": "user", "content": "hi"}],
        },
    }
    sink = mk.records.MemorySink()
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat").respond(json=REPLY)
        mock.post("/api/show").respond(json={})
        new = mk.replay.Replayer(sink=sink).replay_call(span, {})
        del span["attributes"]["gen_ai.provider.name"]
        with pytest.raises(mk.errors.ConfigError, match="names no model"):
            mk.replay.Replayer(sink=sink).replay_call(span, {})
    body = json.loads(chat.calls.last.request.content)
    assert (body["model"], body["options"]["temperature"]) == ("llama3.2:1b", 0.1)
    assert new["attributes"]["hone.models.model_id"] == "ollama:llama3.2:1b"
