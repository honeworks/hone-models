"""AC-7: mk.Prompt with sections -> rendered text correct; section ids, versions and character spans
recorded and consistent with the rendered text."""

import hashlib
import json

import pytest
import respx

import hone_models as mk

pytestmark = pytest.mark.e2e
URL = "http://127.0.0.1:11434"
REPLY = {"model": "m", "message": {"content": '{"title": "x"}'}, "done_reason": "stop"}


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


def test_ac7_prompt_sections_recorded() -> None:
    prompt = song_prompt()
    assert prompt.text == (
        "You write song pitches.\n\nExactly 12 ideas, JSONL.\n\n"
        '{"title": "Cargo Hold Chaos", ...}\n\nTheme: sea shanties'
    )
    sink = mk.records.MemorySink()
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat").respond(json=REPLY)
        r = mk.text("gemma4-12b", sink=sink).complete(prompt)
    assert r.error is None

    sent = json.loads(chat.calls.last.request.content)["messages"]
    assert sent[0] == {"role": "system", "content": "You write song pitches."}
    assert sent[1] == {
        "role": "user",
        "content": 'Exactly 12 ideas, JSONL.\n\n{"title": "Cargo Hold Chaos", ...}\n\nTheme: sea shanties',
    }

    a = sink.spans[0]["attributes"]
    assert a["hone.models.prompt.template_id"] == "song_ideas"
    assert a["hone.models.prompt.template_version"] == "3"
    assert a["hone.models.prompt.variables"] == {"theme": "sea shanties"}
    sections = a["hone.models.prompt.sections"]
    assert [s["id"] for s in sections] == ["system", "rules", "format_example", "brief"]
    assert [s["version"] for s in sections] == ["3", "3", "7", "3"]
    rendered = "\n\n".join(m["content"] for m in a["gen_ai.input.messages"])
    assert rendered == prompt.text
    expected = {
        "system": "You write song pitches.",
        "rules": "Exactly 12 ideas, JSONL.",
        "format_example": '{"title": "Cargo Hold Chaos", ...}',
        "brief": "Theme: sea shanties",
    }
    for s in sections:
        piece = rendered[s["start"] : s["end"]]
        assert piece == expected[s["id"]]
        assert s["sha256"] == hashlib.sha256(piece.encode()).hexdigest()


def test_ac7_system_sections_come_first_and_can_be_chosen() -> None:
    prompt = mk.Prompt(
        "t", "1", {"task": "Do it.", "persona": "You are terse."}, system_sections=("persona",)
    )
    assert prompt.messages() == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Do it."},
    ]
    spans = {s["id"]: (s["start"], s["end"]) for s in prompt.section_records()}
    assert prompt.text[slice(*spans["task"])] == "Do it."
    assert prompt.text[slice(*spans["persona"])] == "You are terse."


def test_ac7_capture_off_keeps_section_spans_but_hashes_variables(isolated) -> None:
    prompt = song_prompt()
    sink = mk.records.SqliteSpanSink(isolated / "spans.db", capture_content=False)
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(json=REPLY)
        mk.text("gemma4-12b", sink=sink).complete(prompt)
    sink.flush()
    a = mk.records.read_spans(isolated / "spans.db")[0]["attributes"]
    assert a["hone.models.prompt.template_id"] == "song_ideas"
    assert a["hone.models.prompt.sections"] == prompt.section_records()
    assert set(a["hone.models.prompt.variables"]) == {"sha256", "len"}
    assert "sea shanties" not in json.dumps(a)
