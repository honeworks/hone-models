"""AC-13: a planted API key never appears in the SQLite file."""

import pytest
import respx

import hone_models as mk

pytestmark = pytest.mark.e2e
KEY = "hm-PLANTED-test-key-0123456789"


def test_ac13_planted_key_never_recorded(isolated, monkeypatch, no_backoff) -> None:
    monkeypatch.setenv("REMOTE_OLLAMA_KEY", KEY)
    reg_file = isolated / "reg.toml"
    reg_file.write_text(
        '[models.remote]\nprovider = "ollama"\nmodel = "llama3.2:1b"\n'
        'base_url = "http://gpu-box:11434"\napi_key_env = "REMOTE_OLLAMA_KEY"\n'
    )
    registry = mk.registry.load([reg_file])
    sink = mk.records.SqliteSpanSink(isolated / "spans.db")
    llm = mk.text("remote", registry=registry, sink=sink)
    with respx.mock(base_url="http://gpu-box:11434") as mock:
        chat = mock.post("/api/chat").respond(
            json={"model": "llama3.2:1b", "message": {"content": f"your key is {KEY}"}, "done_reason": "stop"}
        )
        llm.complete([{"role": "user", "content": f"Remember {KEY}"}], note=KEY)
        mock.post("/api/chat").respond(400, text=f"bad key {KEY}")
        with pytest.raises(mk.errors.ProviderError):
            llm.complete([{"role": "user", "content": "again"}])

    assert chat.calls[0].request.headers["Authorization"] == f"Bearer {KEY}"  # it was really used
    sink.close()
    files = list(isolated.glob("spans.db*"))
    assert files
    for f in files:
        assert KEY.encode() not in f.read_bytes()
    spans = mk.records.read_spans(isolated / "spans.db")
    assert len(spans) == 2
    assert "***" in spans[0]["attributes"]["gen_ai.output.messages"][0]["content"]
    assert spans[1]["status"]["code"] == "error"
    assert "***" in spans[1]["status"]["message"]


def test_ac13_key_in_a_prompt_that_fails_before_sending(isolated, monkeypatch) -> None:
    key = 'sk-PLANTED-"quoted"-0123456789abcdef'
    monkeypatch.setenv("OPENAI_API_KEY", key)
    sink = mk.records.SqliteSpanSink(isolated / "spans.db")
    llm = mk.text("gpt-4.1-mini", sink=sink)
    with respx.mock(assert_all_mocked=True), pytest.raises(mk.errors.ContextOverflow):
        llm.complete([{"role": "user", "content": f"key: {key} " + "x" * 4_000_000}])
    sink.close()
    for f in isolated.glob("spans.db*"):
        assert b"PLANTED" not in f.read_bytes()


def test_ac13_missing_key_fails_before_http(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with respx.mock(assert_all_mocked=True) as mock:
        with pytest.raises(mk.errors.ConfigError, match="OPENAI_API_KEY"):
            mk.text("gpt-4.1-mini", sink=mk.records.NullSink()).complete([{"role": "user", "content": "x"}])
        assert mock.calls.call_count == 0
