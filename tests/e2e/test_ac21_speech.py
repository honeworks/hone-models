"""AC-21: speech - a minutes-long script becomes one WAV file and one recorded span (FakeSpeech)."""

import wave
from pathlib import Path

import pytest

import hone_models as mk
from hone_models.testing import FakeSpeech

pytestmark = pytest.mark.e2e


def test_ac21_long_narration_one_wav_one_span(tmp_path: Path) -> None:
    store = tmp_path / "spans.db"
    sink = mk.records.SqliteSpanSink(store, capture_content=False)
    script = " ".join(f"Point {i} explains one more idea in a single sentence." for i in range(80))
    r = FakeSpeech(sink=sink).synthesize(script, voice="bm_george", out=tmp_path / "n" / "narration.wav")
    sink.close()
    with wave.open(str(r.path), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, r.sample_rate)
        assert wav.getnframes() / r.sample_rate == pytest.approx(r.duration_s)
    assert r.duration_s > 180  # 720 words at 2.5 words/s, plus pauses
    (span,) = mk.records.read_spans(store)
    attrs = span["attributes"]
    assert span["span_id"] == r.span_id
    assert (span["name"], attrs["gen_ai.operation.name"]) == ("hone.models.speech", "speech")
    assert attrs["hone.models.speech.chunks"] > 5
    assert attrs["hone.models.speech.voice"] == "bm_george"
    assert attrs["hone.models.speech.input"]["len"] == len(script)  # content: hashed when capture is off
    assert script not in store.read_bytes().decode("utf-8", "replace")
