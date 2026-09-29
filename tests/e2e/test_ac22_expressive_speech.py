"""AC-22: expressive speech - emotion and intensity per call, recorded on the span; paragraphs stay whole
(FakeSpeech; the real Chatterbox run is tests/gpu/test_ac22_real_expressive_speech.py)."""

from pathlib import Path

import pytest

import hone_models as mk
from hone_models.testing import FakeSpeech

pytestmark = pytest.mark.e2e


def test_ac22_emotion_per_paragraph_recorded_and_paragraphs_whole(tmp_path: Path) -> None:
    store = tmp_path / "spans.db"
    sink = mk.records.SqliteSpanSink(store)
    tts = FakeSpeech(sink=sink, expressive=True)
    lesson = [
        ("curious", "Have you ever wondered why feedback stings? " * 6),
        ("encouraging", "Here is the good news. You can train the first ten seconds."),
    ]
    with tts.session():
        results = [
            tts.synthesize(
                f"{text}\n\nOne more thought.", emotion=emotion, intensity=0.6, out=tmp_path / f"{i}.wav"
            )
            for i, (emotion, text) in enumerate(lesson)
        ]
    assert (tts.loads, tts.frees) == (1, 1)  # one model load for the lesson
    sink.close()
    spans = mk.records.read_spans(store)
    # Match spans to calls by id: both calls can start in the same instant, and then the order of
    # read_spans (start time, then span id) is not the call order.
    by_id = {s["span_id"]: s for s in spans}
    assert sorted(by_id) == sorted(r.span_id for r in results)
    for result, (emotion, _) in zip(results, lesson, strict=True):
        attrs = by_id[result.span_id]["attributes"]
        assert attrs["hone.models.speech.emotion"] == emotion
        assert attrs["hone.models.speech.intensity"] == 0.6
        assert attrs["hone.models.speech.expressive"] is True
        assert attrs["hone.models.speech.paragraphs"] == 2
        assert attrs["hone.models.speech.session"] is True
    first = by_id[results[0].span_id]["attributes"]
    assert first["hone.models.speech.chunks"] == 2  # 270 characters: one chunk, not cut
    assert [c[3:] for c in tts.calls] == [("curious", 0.6), ("encouraging", 0.6)]
    assert mk.speech("chatterbox").expressive
    assert not mk.speech("kokoro-82m").expressive
