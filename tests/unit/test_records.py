import json
import sqlite3
from pathlib import Path

import pytest

from hone_models.records import (
    CONTENT_KEYS,
    JsonlSpanSink,
    MemorySink,
    NullSink,
    SqliteSpanSink,
    add_secret,
    default_sink,
    read_spans,
)
from hone_models.testing import example_span


def span_with(**attrs: object) -> dict:
    s = example_span()
    s["attributes"] = {**s["attributes"], **attrs}
    return s


def test_sqlite_schema_and_meta(tmp_path: Path) -> None:
    sink = SqliteSpanSink(tmp_path / "s.db")
    db = sqlite3.connect(tmp_path / "s.db")
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"meta", "spans", "blobs", "changes"} <= tables
    meta = dict(db.execute("SELECT key, value FROM meta"))
    assert meta["schema"] == "hone-spans"
    assert meta["schema_version"] == "1"
    assert meta["package"] == "hone-models"
    assert meta["created_at"].endswith("Z")
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    sink.close()


def test_sqlite_roundtrip_and_changes(tmp_path: Path) -> None:
    sink = SqliteSpanSink(tmp_path / "s.db")
    sink.emit(span_with(**{"gen_ai.request.model": "m"}))
    sink.flush()
    [got] = read_spans(tmp_path / "s.db")
    assert got == span_with(**{"gen_ai.request.model": "m"})
    db = sqlite3.connect(tmp_path / "s.db")
    assert db.execute("SELECT span_id, op FROM changes").fetchall() == [("00f067aa0ba902b7", "insert")]


def test_large_attribute_goes_to_blobs(tmp_path: Path) -> None:
    big = "x" * (70 * 1024)
    sink = SqliteSpanSink(tmp_path / "s.db")
    sink.emit(span_with(**{"gen_ai.output.messages": [{"content": big}]}))
    raw = sqlite3.connect(tmp_path / "s.db").execute("SELECT attributes FROM spans").fetchone()[0]
    assert '"$blob"' in raw
    assert big not in raw
    [got] = read_spans(tmp_path / "s.db")
    assert got["attributes"]["gen_ai.output.messages"] == [{"content": big}]


def test_capture_off_hashes_content() -> None:
    sink = MemorySink(capture_content=False)
    sink.emit(span_with(**{"gen_ai.input.messages": [{"role": "user", "content": "secret plan"}], "x": 1}))
    stored = sink.spans[0]["attributes"]
    assert set(stored["gen_ai.input.messages"]) == {"sha256", "len"}
    assert len(stored["gen_ai.input.messages"]["sha256"]) == 64
    assert stored["x"] == 1
    assert "secret plan" not in json.dumps(sink.spans)


def test_capture_follows_env_when_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = MemorySink()
    monkeypatch.setenv("HONE_CAPTURE_CONTENT", "0")
    sink.emit(span_with(**{"gen_ai.output.messages": ["hello"]}))
    monkeypatch.setenv("HONE_CAPTURE_CONTENT", "1")
    sink.emit(span_with(**{"gen_ai.output.messages": ["hello"]}))
    assert "sha256" in sink.spans[0]["attributes"]["gen_ai.output.messages"]
    assert sink.spans[1]["attributes"]["gen_ai.output.messages"] == ["hello"]


def test_secrets_are_scrubbed() -> None:
    add_secret("planted-key-1234567")
    add_secret("short")  # too short to register: would mangle ordinary text
    sink = MemorySink()
    sink.emit(
        span_with(
            a="value planted-key-1234567 inside",
            b="Authorization: Bearer abc.def-123",
            c="key sk-abcdefghijklmnopqrstuvwx",
            d="short text",
        )
    )
    attrs = sink.spans[0]["attributes"]
    assert attrs["a"] == "value *** inside"
    assert attrs["b"] == "Authorization: Bearer ***"
    assert attrs["c"] == "key ***"
    assert attrs["d"] == "short text"


def test_sink_failures_are_counted_not_raised(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    sink = SqliteSpanSink(tmp_path / "s.db")
    sink.close()
    sink.emit(example_span())
    sink.emit(example_span())
    assert sink.failures == 2
    assert caplog.text.count("could not record span") == 1


def test_jsonl_and_null_sinks(tmp_path: Path) -> None:
    sink = JsonlSpanSink(tmp_path / "d" / "s.jsonl")
    sink.emit(example_span())
    sink.emit(example_span())
    lines = (tmp_path / "d" / "s.jsonl").read_text().splitlines()
    assert [json.loads(line)["span_id"] for line in lines] == ["00f067aa0ba902b7"] * 2
    null = NullSink()
    null.emit(example_span())
    null.flush()
    null.close()
    assert null.failures == 0


def test_default_sink_uses_hone_home(isolated: Path) -> None:
    sink = default_sink()
    assert isinstance(sink, SqliteSpanSink)
    assert sink.path == (isolated / ".hone" / "models" / "spans.db").resolve()
    assert default_sink() is sink


def test_planted_secret_never_reaches_sqlite_bytes(tmp_path: Path) -> None:
    key = 'sk-test-PLANTED"quote\\1234567890'
    add_secret(key)
    sink = SqliteSpanSink(tmp_path / "s.db")
    sink.emit(span_with(small=f"key={key}", big=[key + "y" * (70 * 1024)]))
    sink.close()
    for f in tmp_path.glob("s.db*"):
        assert b"PLANTED" not in f.read_bytes()


def test_every_content_key_is_hashed_in_sqlite(tmp_path: Path) -> None:
    sink = SqliteSpanSink(tmp_path / "s.db", capture_content=False)
    sink.emit(span_with(**{k: f"content of {k}" for k in CONTENT_KEYS}))
    [got] = read_spans(tmp_path / "s.db")
    for key in CONTENT_KEYS:
        assert set(got["attributes"][key]) == {"sha256", "len"}
    assert "content of" not in json.dumps(got)


def test_default_sink_falls_back_when_store_unwritable(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    (isolated / "file").write_text("not a directory")
    monkeypatch.setenv("HONE_HOME", str(isolated / "file"))
    sink = default_sink()
    assert isinstance(sink, NullSink)
    assert "cannot open span store" in caplog.text
