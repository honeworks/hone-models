"""Testing your own code with the public `FakeOllama`: no models, no GPU, fast and deterministic.

What: unit tests for code that calls hone-models: check what it sends, script the model's answers
    (including failures) and check the contract of your own record sink.

How: `with FakeOllama() as server:` starts a local fake Ollama server and points `OLLAMA_HOST` at it,
    so the code under test needs no change. Chat replies echo the last user message, or return a sample
    object when a JSON Schema is sent; embeddings are stable hash vectors. `server.requests` holds every
    request (`path`, `body`); `server.queue(path, *answers)` scripts the next answers for one path (a
    dict is merged over the default reply, an int is an HTTP error status). When the number or order of
    calls varies, `FakeOllama(responder=fn)` answers from the request instead: `fn(path, body)` returns a
    dict merged over the default reply, or `None` for the default. Pass
    `sink=mk.records.MemorySink()` to assert on the recorded spans. As a pytest fixture:

        @pytest.fixture
        def ollama():
            with FakeOllama() as server:
                yield server

    `hone_models.testing.check_record_sink(sink, read_back)` checks a custom `RecordSink`.

Why: tests against a real model are slow, need a GPU and give different answers each run; keep them for
    a few end-to-end checks (run under the shared GPU lock). Pitfall: without an explicit `sink=` calls
    are recorded to `${HONE_HOME:-.hone}/models/spans.db`; point `HONE_HOME` at a temporary folder in
    tests or pass a `MemorySink`.

Run with pytest (`pytest examples/testing_with_fake_ollama.py`) or as a script, which runs pytest.
"""

import pytest
from pydantic import BaseModel

import hone_models as mk
from hone_models.testing import FakeOllama, check_record_sink


# The code under test: yours.
def summarize(llm: mk.TextClient, text: str) -> str | None:
    """One-line summary, or None when the model gave no usable answer."""
    r = llm.complete(
        [{"role": "system", "content": "Summarize in one line."}, {"role": "user", "content": text}],
        max_tokens=60,
    )
    return None if r.error else r.text


# The tests.
def test_summarize_sends_the_text_and_returns_the_reply() -> None:
    sink = mk.records.MemorySink()
    with FakeOllama() as server:
        summary = summarize(mk.text("gemma4-12b", sink=sink), "A long story about the sea.")
    assert summary == "echo: A long story about the sea."
    sent = server.requests[-1]["body"]
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["options"]["num_predict"] == 60
    assert sink.spans[-1]["status"]["code"] == "ok"


def test_summarize_returns_none_for_an_empty_answer() -> None:
    with FakeOllama() as server:
        server.queue("/api/chat", {"message": {"role": "assistant", "content": ""}})
        assert summarize(mk.text("gemma4-12b", sink=mk.records.MemorySink()), "text") is None


def test_server_errors_reach_the_caller() -> None:
    with FakeOllama() as server:
        server.queue("/api/chat", 404)
        with pytest.raises(mk.errors.ProviderError) as info:
            summarize(mk.text("gemma4-12b", sink=mk.records.MemorySink()), "text")
    assert info.value.status == 404


def answer_by_schema(path: str, body: dict) -> dict | None:
    """A responder: a strict gate needs a real answer, so give one for the schema titled `Rating`."""
    if path == "/api/chat" and (body.get("format") or {}).get("title") == "Rating":
        return {"message": {"role": "assistant", "content": '{"stars": 4, "why": "clear and short"}'}}
    return None  # everything else: the default reply


class Rating(BaseModel):
    stars: int
    why: str


def test_a_responder_answers_by_schema_however_many_calls_come() -> None:
    with FakeOllama(responder=answer_by_schema):
        llm = mk.text("gemma4-12b", sink=mk.records.MemorySink())
        for _ in range(3):  # the order and number of calls do not matter
            assert llm.complete([{"role": "user", "content": "rate it"}], schema=Rating).parsed == Rating(
                stars=4, why="clear and short"
            )
        assert llm.complete([{"role": "user", "content": "hi"}]).text == "echo: hi"


class ListSink:
    """A custom record sink (e.g. to forward spans elsewhere): `emit` and `flush`, never raising."""

    def __init__(self) -> None:
        self.spans: list[dict] = []

    def emit(self, span: dict) -> None:
        self.spans.append(span)

    def flush(self) -> None:
        pass

    def read(self) -> list[dict]:
        return self.spans


def test_custom_sink_meets_the_record_sink_contract() -> None:
    sink = ListSink()
    check_record_sink(sink, sink.read)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
