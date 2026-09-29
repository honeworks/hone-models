"""The transcriber (`mk.transcriber`) and `FakeTranscriber`: checks, spans, leases, sessions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import hone_models as mk
from hone_models.errors import CapabilityError, ConfigError, ProviderError
from hone_models.speech import write_wav
from hone_models.testing import FakeTranscriber
from media_fixtures import LeaseRecorder
from whisper_fakes import install, registry


@pytest.fixture
def take(tmp_path: Path) -> Path:
    path = tmp_path / "take.wav"
    write_wav(path, b"\x00\x00" * 1600, 16000)
    return path


def test_fake_hears_its_text_with_evenly_spaced_words(take: Path) -> None:
    stt = FakeTranscriber(text="Hello there.\n\nSecond line here.")
    t = stt.transcribe(take, prompt="Hello")
    assert t.text == "Hello there. Second line here."
    assert [s.text for s in t.segments] == ["Hello there.", "Second line here."]
    assert [(w.text, w.start_s, w.end_s) for w in t.words] == [
        ("Hello", 0.0, 0.4),
        ("there.", 0.4, 0.8),
        ("Second", 0.8, 1.2),
        ("line", 1.2, 1.6),
        ("here.", 1.6, 2.0),
    ]
    assert t.segments[1].words == t.words[2:]
    assert (t.language, t.duration_s, t.model) == ("en", 2.0, "fake-transcriber")
    assert stt.calls == [(take, None, "Hello", True)]
    stt.text = "Other words."
    no_words = stt.transcribe(str(take), language="de", words=False)
    assert (no_words.text, no_words.language, no_words.words, no_words.segments[0].words) == (
        "Other words.",
        "de",
        [],
        [],
    )


def test_span_records_audio_and_result(take: Path) -> None:
    sink = mk.records.MemorySink()
    stt = FakeTranscriber(sink=sink)
    t = stt.transcribe(take, language="en", prompt="lyrics here", trace={"hone.run_id": "run-1"})
    (span,) = sink.spans
    attrs = span["attributes"]
    assert (span["name"], span["span_id"], span["status"]["code"]) == (
        "hone.models.transcribe",
        t.span_id,
        "ok",
    )
    assert attrs["gen_ai.operation.name"] == "transcription"
    assert attrs["gen_ai.provider.name"] == "faster_whisper"
    assert attrs["hone.run_id"] == "run-1"
    data = take.read_bytes()
    assert attrs["hone.models.transcribe.audio"] == {
        "path": str(take),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    assert attrs["hone.models.transcribe.language"] == "en"
    assert attrs["hone.models.transcribe.duration_s"] == t.duration_s
    assert attrs["hone.models.transcribe.words_count"] == 4
    assert attrs["hone.models.transcribe.text"] == "Hello there, and welcome."
    assert attrs["hone.models.transcribe.words"][0] == {
        "text": "Hello",
        "start_s": 0.0,
        "end_s": 0.4,
        "probability": 1.0,
    }
    assert attrs["hone.models.transcribe.prompt"] == "lyrics here"
    assert attrs["hone.models.transcribe.session"] is False
    assert attrs["hone.models.transcribe.loaded"] is True


def test_prompt_is_absent_from_the_span_when_not_given(take: Path) -> None:
    sink = mk.records.MemorySink()
    FakeTranscriber(sink=sink).transcribe(take)
    assert "hone.models.transcribe.prompt" not in sink.spans[0]["attributes"]


def test_bad_calls_raise_before_any_work(tmp_path: Path, take: Path) -> None:
    stt = FakeTranscriber()
    with pytest.raises(ConfigError, match="does not exist"):
        stt.transcribe(tmp_path / "missing.wav")
    with pytest.raises(ConfigError, match="timeout_s"):
        stt.transcribe(take, timeout_s=0)
    cpu = FakeTranscriber.like("test-whisper-cpu", registry=registry())
    with pytest.raises(CapabilityError, match="words=False"):
        cpu.transcribe(take)
    assert cpu.transcribe(take, words=False).words == []
    assert stt.calls == []
    assert (stt.loads, stt.sink.spans) == (0, [])  # type: ignore[attr-defined]


def test_factory_checks_the_kind() -> None:
    reg = registry()
    stt = mk.transcriber("test-whisper", registry=reg, sink=mk.records.NullSink())
    assert (stt.model_id, stt.config.local) == ("test-whisper", True)
    with pytest.raises(ConfigError, match=r"mk\.transcriber\(\) takes transcription models"):
        mk.transcriber("kokoro-82m", registry=reg)
    with pytest.raises(ConfigError, match="transcription models"):
        FakeTranscriber.like("kokoro-82m")
    fake = FakeTranscriber.like("test-whisper", registry=reg)
    assert (fake.model_id, fake.config.capabilities.vram_gb) == ("test-whisper", 2.0)


def test_plain_calls_load_and_free_each_time_sessions_once(take: Path) -> None:
    sink = mk.records.MemorySink()
    stt = FakeTranscriber(sink=sink)
    stt.transcribe(take)
    stt.transcribe(take)
    assert (stt.loads, stt.frees) == (2, 2)
    with stt.session() as held:
        assert held is stt
        held.transcribe(take)
        held.transcribe(take)
        assert (stt.loads, stt.frees) == (3, 2)
        with pytest.raises(ConfigError, match="already open"), stt.session():
            pass
    assert (stt.loads, stt.frees) == (3, 3)
    loaded = [s["attributes"]["hone.models.transcribe.loaded"] for s in sink.spans]
    in_session = [s["attributes"]["hone.models.transcribe.session"] for s in sink.spans]
    assert loaded == [True, True, True, False]
    assert in_session == [False, False, True, True]


def test_session_frees_on_an_exception(take: Path) -> None:
    stt = FakeTranscriber()

    def fail_in_a_session() -> None:
        with stt.session():
            stt.transcribe(take)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        fail_in_a_session()
    assert (stt.loads, stt.frees) == (1, 1)
    with stt.session():  # a new session opens after the failed one
        pass


def test_real_client_leases_per_call_or_once_per_session(take: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(monkeypatch)
    stt = mk.transcriber("test-whisper", registry=registry(), sink=mk.records.MemorySink())
    leases = LeaseRecorder()
    stt.gpu = leases
    stt.transcribe(take)
    assert leases.calls == [("test-whisper", 2.0)]
    with stt.session():
        assert leases.held == 1
        stt.transcribe(take)
        stt.transcribe(take)
    assert leases.calls == [("test-whisper", 2.0)] * 2
    assert (len(fake.loads), fake.unloads, leases.held) == (2, 2, 0)


def test_a_failing_model_records_an_error_span_and_frees(take: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(monkeypatch)
    sink = mk.records.MemorySink()
    stt = mk.transcriber("test-whisper", registry=registry(), sink=sink)
    stt.gpu = LeaseRecorder()
    fake.lines = [(" ", 0.0, 1.0)]  # a segment without words: the fake divides by zero
    with pytest.raises(ProviderError, match="faster-whisper transcription failed"):
        stt.transcribe(take)
    assert sink.spans[0]["status"]["code"] == "error"
    assert fake.unloads == 1


def test_long_word_lists_go_to_the_blob_table(tmp_path: Path, take: Path) -> None:
    import sqlite3  # noqa: PLC0415

    store = tmp_path / "spans.db"
    sink = mk.records.SqliteSpanSink(store)
    text = " ".join(f"word{i}" for i in range(2000))
    t = FakeTranscriber(text=text, sink=sink).transcribe(take)
    sink.close()
    with sqlite3.connect(store) as db:
        assert db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] >= 1
    (span,) = mk.records.read_spans(store)
    assert len(span["attributes"]["hone.models.transcribe.words"]) == len(t.words) == 2000
