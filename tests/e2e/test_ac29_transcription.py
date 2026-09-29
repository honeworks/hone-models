"""AC-29: transcription with the fake `faster_whisper` module, content capture on and off - words with
times on the result, one span, text and words hashed with capture off."""

from pathlib import Path

import pytest

import hone_models as mk
from hone_models.speech import write_wav
from media_fixtures import LeaseRecorder
from whisper_fakes import install, registry

pytestmark = pytest.mark.e2e
HEARD = "Hello there, and welcome. This is the second line."


@pytest.mark.parametrize("capture", [True, False])
def test_ac29_words_with_times_and_one_span(
    capture: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    take = tmp_path / "take.wav"
    write_wav(take, b"\x00\x00" * 16000, 16000)
    store = tmp_path / "spans.db"
    sink = mk.records.SqliteSpanSink(store, capture_content=capture)
    stt = mk.transcriber("test-whisper", registry=registry(), sink=sink)
    leases = LeaseRecorder()
    stt.gpu = leases

    t = stt.transcribe(take, language="en", prompt="welcome")
    sink.close()

    assert t.text == HEARD
    assert [w.text for w in t.words] == HEARD.split()
    starts = [w.start_s for w in t.words]
    assert starts == sorted(starts)
    assert all(w.end_s > w.start_s for w in t.words)
    assert t.segments[1].words[0] == mk.Word("This", 1.9, 2.26, 0.9)
    assert leases.calls == [("test-whisper", 2.0)]

    (span,) = mk.records.read_spans(store)
    attrs = span["attributes"]
    assert (span["name"], span["span_id"]) == ("hone.models.transcribe", t.span_id)
    assert attrs["hone.models.transcribe.words_count"] == len(HEARD.split())
    assert attrs["hone.models.transcribe.duration_s"] == 4.0
    assert attrs["hone.models.transcribe.audio"]["bytes"] == take.stat().st_size
    stored = store.read_bytes().decode("utf-8", "replace")
    if capture:
        assert attrs["hone.models.transcribe.text"] == HEARD
        assert attrs["hone.models.transcribe.words"][1]["text"] == "there,"
        assert "second line" in stored
    else:
        for key in ("text", "words", "prompt"):
            assert set(attrs[f"hone.models.transcribe.{key}"]) == {"sha256", "len"}
        assert "second line" not in stored
        assert "welcome" not in stored
