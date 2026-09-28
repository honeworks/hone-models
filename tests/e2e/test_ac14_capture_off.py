"""AC-14: with content capture off, messages are stored as hashes + lengths only."""

import json

import pytest
import respx

import hone_models as mk

pytestmark = pytest.mark.e2e
URL = "http://127.0.0.1:11434"
PRIVATE_TEXT = "the launch codes are 0000"


def call(sink) -> None:
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(
            json={"model": "m", "message": {"content": "noted"}, "done_reason": "stop"}
        )
        mk.text("gemma4-12b", sink=sink).complete([{"role": "user", "content": PRIVATE_TEXT}])


@pytest.mark.parametrize("how", ["env", "sink"])
def test_ac14_capture_off(isolated, monkeypatch, how) -> None:
    if how == "env":
        monkeypatch.setenv("HONE_CAPTURE_CONTENT", "0")
        sink = mk.records.SqliteSpanSink(isolated / "spans.db")
    else:
        sink = mk.records.SqliteSpanSink(isolated / "spans.db", capture_content=False)
    call(sink)
    sink.close()
    [span] = mk.records.read_spans(isolated / "spans.db")
    for key in ("gen_ai.input.messages", "gen_ai.output.messages"):
        stored = span["attributes"][key]
        assert set(stored) == {"sha256", "len"}
        assert len(stored["sha256"]) == 64
        assert stored["len"] > 0
    assert span["attributes"]["gen_ai.request.model"] == "gemma4-12b:latest"
    for f in isolated.glob("spans.db*"):
        assert PRIVATE_TEXT.encode() not in f.read_bytes()


def test_ac14_capture_on_by_default(isolated) -> None:
    sink = mk.records.SqliteSpanSink(isolated / "spans.db")
    call(sink)
    [span] = mk.records.read_spans(isolated / "spans.db")
    assert json.dumps(span["attributes"]["gen_ai.input.messages"]).count(PRIVATE_TEXT) == 1


def test_ac14_capture_off_hides_output_in_retries_and_errors(isolated) -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    sink = mk.records.SqliteSpanSink(isolated / "spans.db", capture_content=False)
    llm = mk.text("gemma4-12b", sink=sink)
    bad = {"model": "m", "message": {"content": f'{{"n": "{PRIVATE_TEXT}"}}'}, "done_reason": "stop"}
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(json=bad)
        r = llm.complete([{"role": "user", "content": "n?"}], schema=schema)
        mock.post("/api/chat").respond(400, text=f"cannot process: {PRIVATE_TEXT}")
        with pytest.raises(mk.errors.ProviderError):
            llm.complete([{"role": "user", "content": "n?"}])
    assert r.structured_path == "failed"
    sink.close()
    for f in isolated.glob("spans.db*"):
        assert PRIVATE_TEXT.encode() not in f.read_bytes()
    spans = mk.records.read_spans(isolated / "spans.db")
    assert all(e["attributes"]["reason"].keys() == {"sha256", "len"} for e in spans[0]["events"])
