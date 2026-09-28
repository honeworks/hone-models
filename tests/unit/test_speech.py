import wave
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import hone_models as mk
from hone_models.errors import ConfigError, ProviderError
from hone_models.providers import SPEECH
from hone_models.speech import EMOTIONS, join_paragraphs, join_pcm, split_paragraphs, split_text, write_wav
from hone_models.testing import FakeSpeech


def frames(path: Path) -> tuple[int, int, bytes]:
    with wave.open(str(path), "rb") as wav:
        return wav.getframerate(), wav.getnframes(), wav.readframes(wav.getnframes())


def test_split_keeps_short_text_whole() -> None:
    assert split_text("  Hello   there.\nHow are you?  ") == ["Hello there. How are you?"]
    assert split_text(" \n ") == []


def test_split_packs_sentences_into_chunks() -> None:
    text = "One two three. Four five six! Seven eight nine? Ten."
    chunks = split_text(text, max_chars=30)
    assert chunks == ["One two three. Four five six!", "Seven eight nine? Ten."]
    assert all(len(c) <= 30 for c in chunks)


def test_split_cuts_long_sentences_at_spaces_and_breaks_paragraphs() -> None:
    long = " ".join(["word"] * 50)  # 249 characters, no sentence end
    chunks = split_text(long, max_chars=40)
    assert all(len(c) <= 40 for c in chunks)
    assert " ".join(chunks) == long
    assert split_text("x" * 25, max_chars=10) == ["x" * 10, "x" * 10, "x" * 5]
    assert split_text("Title\n\nBody text", max_chars=10) == ["Title", "Body text"]


def test_split_keeps_every_word_of_minutes_long_text() -> None:
    text = " ".join(f"Sentence number {i} says something useful." for i in range(300))
    chunks = split_text(text)
    assert len(chunks) > 20
    assert all(len(c) <= 400 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_join_pcm_puts_silence_between_chunks(tmp_path: Path) -> None:
    pcm = join_pcm([b"\x01\x00" * 10, b"\x02\x00" * 10], sample_rate=100, pause_s=0.5)
    assert pcm == b"\x01\x00" * 10 + b"\x00\x00" * 50 + b"\x02\x00" * 10
    duration = write_wav(tmp_path / "a" / "b.wav", pcm, 100)
    assert duration == pytest.approx(0.7)
    assert frames(tmp_path / "a" / "b.wav")[:2] == (100, 70)


def test_fake_speech_writes_plausible_silent_wav(tmp_path: Path) -> None:
    tts = FakeSpeech()
    text = " ".join(["word"] * 25) + "."
    r = tts.synthesize(text, out=tmp_path / "out.wav")
    rate, n, data = frames(r.path)
    assert (r.path, r.sample_rate, rate, r.voice, r.model) == (
        tmp_path / "out.wav",
        24000,
        24000,
        "af_heart",
        "fake-speech",
    )
    assert r.duration_s == pytest.approx(10.0)
    assert n == 240000
    assert set(data) == {0}
    assert tts.calls == [(text, "af_heart", 1.0, None, None)]
    fast = tts.synthesize(text, voice="am_michael", speed=2.0, out=tmp_path / "fast.wav")
    assert fast.duration_s == pytest.approx(5.0)


def test_speech_span_records_text_as_content(tmp_path: Path) -> None:
    sink = mk.records.MemorySink(capture_content=False)
    tts = FakeSpeech(sink=sink)
    r = tts.synthesize(
        "Hello there. " * 50, voice="am_michael", out=tmp_path / "x.wav", trace={"hone.run_id": "run-1"}
    )
    span = sink.spans[0]
    attrs = span["attributes"]
    assert span["name"] == "hone.models.speech"
    assert span["span_id"] == r.span_id
    assert span["status"]["code"] == "ok"
    assert attrs["gen_ai.operation.name"] == "speech"
    assert attrs["hone.run_id"] == "run-1"
    assert attrs["hone.models.speech.voice"] == "am_michael"
    assert attrs["hone.models.speech.chunks"] == 2
    assert attrs["hone.models.speech.duration_s"] == pytest.approx(r.duration_s, abs=0.001)
    assert attrs["hone.models.speech.sample_rate"] == 24000
    assert set(attrs["hone.models.speech.input"]) == {"sha256", "len"}  # content capture off


def test_bad_arguments_raise_config_error(tmp_path: Path) -> None:
    tts = FakeSpeech()
    with pytest.raises(ConfigError, match="unknown voice"):
        tts.synthesize("Hi.", voice="nope", out=tmp_path / "x.wav")
    with pytest.raises(ConfigError, match="speed"):
        tts.synthesize("Hi.", speed=5, out=tmp_path / "x.wav")
    with pytest.raises(ConfigError, match="empty"):
        tts.synthesize("  ", out=tmp_path / "x.wav")
    with pytest.raises(ConfigError, match="no voices"):
        FakeSpeech(voices=()).synthesize("Hi.", out=tmp_path / "x.wav")
    assert FakeSpeech(voices=()).synthesize("Hi.", voice="any", out=tmp_path / "y.wav").voice == "any"


def test_registry_entry_and_kind_checks() -> None:
    tts = mk.speech("kokoro-82m", sink=mk.records.NullSink())
    assert tts.model_id == "kokoro-82m"
    assert tts.voices[0] == "af_heart"
    assert {"am_michael", "bf_emma", "bm_george"} <= set(tts.voices)
    assert tts.config.local
    with pytest.raises(ConfigError, match="not a speech model"):
        mk.speech("gemma4-12b")
    with pytest.raises(ConfigError, match=r"mk\.speech"):
        mk.text("kokoro-82m")
    with pytest.raises(ConfigError, match=r"mk\.speech"):
        mk.decision("kokoro-82m")


def test_real_client_takes_a_gpu_lease_and_reports_backend_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leases: list[tuple[str, float]] = []

    class Lease:
        def lease(self, name: str, vram_gb: float, **_: object):
            leases.append((name, vram_gb))
            return mk.gpu.NullGpuLease().lease(name, vram_gb)

    def broken(*_: object) -> tuple[int, list[bytes]]:
        raise ProviderError("backend down")

    monkeypatch.setitem(SPEECH, "kokoro", broken)
    sink = mk.records.MemorySink()
    tts = mk.speech("kokoro-82m", sink=sink)
    tts.gpu = Lease()
    with pytest.raises(ProviderError, match="backend down"):
        tts.synthesize("Hello.", out=tmp_path / "x.wav")
    assert leases == [("kokoro-82m", 1.0)]
    assert sink.spans[0]["status"]["code"] == "error"
    assert not (tmp_path / "x.wav").exists()


def test_kokoro_without_the_extra_names_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from hone_models.providers import kokoro  # noqa: PLC0415

    def missing(name: str) -> None:
        raise ImportError(name)

    cfg = mk.registry.load().get("kokoro-82m")
    monkeypatch.setattr(kokoro.importlib, "import_module", missing)
    with pytest.raises(ConfigError, match=r"hone-models\[speech\]"):
        kokoro.Engine(cfg)


def test_paragraphs_are_never_merged_and_fit_whole() -> None:
    one = "First idea here. It has two sentences."
    two = "Second idea.\nStill the same paragraph."
    assert split_paragraphs(f"{one}\n\n{two}\n \n") == [[one], ["Second idea. Still the same paragraph."]]
    assert split_text(f"{one}\n\n{two}") == [one, "Second idea. Still the same paragraph."]


def test_long_paragraph_is_split_on_sentences_into_even_chunks() -> None:
    sentences = [f"Sentence {i} is about this long." for i in range(10)]  # 30 characters each
    (chunks,) = split_paragraphs(" ".join(sentences), max_chars=200)
    assert [len(c) for c in chunks] == [154, 154]  # 309 characters: two even halves, not 185 + 123
    assert " ".join(chunks) == " ".join(sentences)
    assert all(c.endswith(".") for c in chunks)  # cut between sentences only


def test_join_paragraphs_uses_a_longer_pause_between_paragraphs() -> None:
    a, b, c = b"\x01\x00", b"\x02\x00", b"\x03\x00"
    pcm = join_paragraphs([a, b, c], [2, 1], sample_rate=20)  # 0.25 s = 5 frames, 0.6 s = 12 frames
    assert pcm == a + b"\x00\x00" * 5 + b + b"\x00\x00" * 12 + c


def test_fake_speech_takes_emotion_and_intensity_and_records_them(tmp_path: Path) -> None:
    sink = mk.records.MemorySink()
    tts = FakeSpeech(sink=sink, expressive=True)
    text = "You did it.\n\nThat was the hardest part."
    r = tts.synthesize(text, emotion="encouraging", intensity=0.7, out=tmp_path / "x.wav")
    assert tts.expressive
    assert tts.emotions == list(EMOTIONS)
    assert tts.calls == [("You did it. That was the hardest part.", "af_heart", 1.0, "encouraging", 0.7)]
    attrs = sink.spans[0]["attributes"]
    assert attrs["hone.models.speech.emotion"] == "encouraging"
    assert attrs["hone.models.speech.intensity"] == 0.7
    assert attrs["hone.models.speech.expressive"] is True
    assert (attrs["hone.models.speech.paragraphs"], attrs["hone.models.speech.chunks"]) == (2, 2)
    words = 8 / 2.5
    assert r.duration_s == pytest.approx(words + 0.6, abs=0.01)  # one paragraph pause


def test_missing_emotion_is_absent_from_the_span_not_zero(tmp_path: Path) -> None:
    sink = mk.records.MemorySink()
    FakeSpeech(sink=sink).synthesize("Hi.", intensity=0.0, out=tmp_path / "x.wav")
    attrs = sink.spans[0]["attributes"]
    assert "hone.models.speech.emotion" not in attrs
    assert attrs["hone.models.speech.intensity"] == 0.0
    assert attrs["hone.models.speech.expressive"] is False


def test_bad_emotion_or_intensity_raise_config_error(tmp_path: Path) -> None:
    tts = FakeSpeech()
    with pytest.raises(ConfigError, match="unknown emotion 'angry'"):
        tts.synthesize("Hi.", emotion="angry", out=tmp_path / "x.wav")
    for bad in (-0.1, 1.5):
        with pytest.raises(ConfigError, match="intensity"):
            tts.synthesize("Hi.", intensity=bad, out=tmp_path / "x.wav")
    assert tts.calls == []


def test_kokoro_accepts_and_ignores_emotion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hone_models.providers import kokoro  # noqa: PLC0415

    torch: Any = pytest.importorskip("torch")
    seen: list[tuple[object, ...]] = []

    class Pipeline:
        def __init__(self, **_: object) -> None:
            pass

        def __call__(self, chunk: str, voice: str, speed: float, split_pattern: None) -> list[Any]:
            seen.append((chunk, voice, speed))
            return [SimpleNamespace(audio=torch.full((10,), 0.1))]

    model = SimpleNamespace(to=lambda _: SimpleNamespace(eval=lambda: "model"))
    library = SimpleNamespace(KModel=lambda repo_id: model, KPipeline=Pipeline)
    real = kokoro._import
    monkeypatch.setattr(kokoro, "_import", lambda name: library if name == "kokoro" else real(name))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # never touch the shared GPU here
    sink = mk.records.MemorySink()
    tts = mk.speech("kokoro-82m", sink=sink)
    tts.gpu = mk.gpu.NullGpuLease()
    assert not tts.expressive
    tts.synthesize("Hello.", voice="am_michael", emotion="excited", intensity=1.0, out=tmp_path / "x.wav")
    assert seen == [("Hello.", "am_michael", 1.0)]  # the emotion never reaches Kokoro
    attrs = sink.spans[0]["attributes"]
    assert (attrs["hone.models.speech.emotion"], attrs["hone.models.speech.expressive"]) == ("excited", False)


def test_chatterbox_registry_entry() -> None:
    tts = mk.speech("chatterbox", sink=mk.records.NullSink())
    assert tts.expressive
    assert tts.voices[:2] == ["warm_female", "warm_male"]
    assert tts.max_chunk_chars == 400
    assert tts.config.capabilities.license == "MIT"
    assert tts.config.local
    assert mk.registry.load().select({"expressive": True}, kind="speech").id == "chatterbox"


def test_plain_calls_load_and_free_the_model_each_time(tmp_path: Path) -> None:
    sink = mk.records.MemorySink()
    tts = FakeSpeech(sink=sink)
    for i in range(3):
        tts.synthesize("Hi.", out=tmp_path / f"{i}.wav")
    assert (tts.loads, tts.frees) == (3, 3)
    assert [s["attributes"]["hone.models.speech.session"] for s in sink.spans] == [False] * 3
    assert [s["attributes"]["hone.models.speech.loaded"] for s in sink.spans] == [True] * 3


def test_session_loads_once_frees_once_and_holds_one_lease(tmp_path: Path) -> None:
    events: list[str] = []

    class Lease:
        @contextmanager
        def lease(self, name: str, vram_gb: float, **_: object) -> Generator[None]:
            events.append(f"lease {name} {vram_gb}")
            yield
            events.append("release")

    sink = mk.records.MemorySink()
    tts = FakeSpeech(sink=sink)
    tts.gpu = Lease()
    with tts.session() as same:
        assert same is tts
        for i, emotion in enumerate(("curious", "warm", "excited")):
            tts.synthesize("One line.", emotion=emotion, out=tmp_path / f"{i}.wav")
            events.append("call")
        assert (tts.loads, tts.frees) == (1, 0)  # still loaded inside the block
    assert (tts.loads, tts.frees) == (1, 1)
    assert events == ["lease fake-speech 1.0", "call", "call", "call", "release"]
    attrs = [s["attributes"] for s in sink.spans]
    assert [a["hone.models.speech.session"] for a in attrs] == [True] * 3
    assert [a["hone.models.speech.loaded"] for a in attrs] == [True, False, False]
    tts.synthesize("After.", out=tmp_path / "after.wav")  # back to load-and-free
    assert (tts.loads, tts.frees) == (2, 2)
    assert events[-2:] == ["lease fake-speech 1.0", "release"]


def test_session_frees_on_error_and_is_not_reentrant(tmp_path: Path) -> None:
    tts = FakeSpeech()
    with tts.session():
        tts.synthesize("Hi.", out=tmp_path / "a.wav")
        with pytest.raises(ConfigError, match="unknown voice"):
            tts.synthesize("Hi.", voice="nobody", out=tmp_path / "b.wav")
    assert (tts.loads, tts.frees) == (1, 1)

    def fail_inside() -> None:
        with tts.session():
            tts.synthesize("Hi.", out=tmp_path / "a.wav")
            raise RuntimeError("the app failed")

    with pytest.raises(RuntimeError, match="the app failed"):
        fail_inside()
    assert (tts.loads, tts.frees) == (2, 2)  # freed although the block raised
    with tts.session(), pytest.raises(ConfigError, match="already open"), tts.session():
        pass
    assert (tts.loads, tts.frees) == (2, 2)  # a session with no calls never loads


def test_english_kokoro_voice_without_the_spacy_model_fails_before_the_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Change 0008: misaki would pip-install en_core_web_sm at run time; say what to add instead."""
    from hone_models.providers import kokoro  # noqa: PLC0415

    real = kokoro.importlib.util.find_spec
    # misaki is "installed" (the speech extra) and the spaCy model is not, with or without the extra here
    fake_specs = {"en_core_web_sm": None, "misaki": SimpleNamespace(name="misaki")}

    def find_spec(name: str) -> object:
        return fake_specs[name] if name in fake_specs else real(name)

    monkeypatch.setattr(kokoro.importlib.util, "find_spec", find_spec)
    leases: list[str] = []

    class Lease:
        def lease(self, name: str, vram_gb: float, **_: object):
            leases.append(name)
            return mk.gpu.NullGpuLease().lease(name, vram_gb)

    tts = mk.speech("kokoro-82m", sink=mk.records.NullSink())
    tts.gpu = Lease()
    with pytest.raises(ConfigError, match=r"en_core_web_sm.*uv pip install https://github.com/explosion"):
        tts.synthesize("Hello.", voice="am_michael", out=tmp_path / "x.wav")
    assert leases == []
    kokoro.check_ready("jf_alpha")  # a Japanese voice does not need it
    FakeSpeech().synthesize("Hello.", out=tmp_path / "fake.wav")  # the fake needs nothing installed


def test_segments_give_each_chunk_its_place_in_the_wav(tmp_path: Path) -> None:
    """Change 0009: timings from the join points; one segment per sentence with `by_sentence`."""
    sink = mk.records.MemorySink()
    tts = FakeSpeech(sample_rate=1000, sink=sink)
    text = "One two three four five. Six seven.\n\nEight nine ten."
    r = tts.synthesize(text, out=tmp_path / "a.wav")
    assert [s.text for s in r.segments] == ["One two three four five. Six seven.", "Eight nine ten."]
    assert r.segments[0].start_s == 0.0
    assert r.segments[0].end_s == 2.8  # 7 words / 2.5 words per second
    assert r.segments[1].start_s == round(2.8 + 0.6, 3)  # the paragraph pause
    assert r.segments[1].end_s == r.duration_s

    r = tts.synthesize(text, out=tmp_path / "b.wav", by_sentence=True)
    assert [(s.text, s.start_s, s.end_s) for s in r.segments] == [
        ("One two three four five.", 0.0, 2.0),
        ("Six seven.", 2.25, 3.05),  # after the 0.25 s pause between chunks
        ("Eight nine ten.", 3.65, 4.85),
    ]
    assert r.duration_s == 4.85
    assert sink.spans[-1]["attributes"]["hone.models.speech.chunks"] == 3


def test_expressive_fake_accepts_chatterbox_voices_and_like_copies_an_entry(tmp_path: Path) -> None:
    """Change 0012: data-stories' `warm_male` on `FakeSpeech(expressive=True)` raised unknown voice."""
    chatterbox = mk.registry.load().get("chatterbox")
    fake = FakeSpeech(expressive=True)
    assert fake.synthesize("Hi.", voice="warm_male", out=tmp_path / "a.wav").voice == "warm_male"
    assert fake.voices[0] == "af_heart"  # tests written before keep their default and Kokoro names
    assert set(chatterbox.capabilities.voices or []) <= set(fake.voices)

    like = FakeSpeech.like("chatterbox")
    assert (like.model_id, like.voices, like.expressive) == (
        "chatterbox",
        chatterbox.capabilities.voices,
        True,
    )
    r = like.synthesize("Hi there.", emotion="warm", out=tmp_path / "b.wav")
    assert (r.voice, r.model) == ("warm_female", "chatterbox")
    assert like.loads == 1
    kokoro = FakeSpeech.like("kokoro-82m")
    assert not kokoro.expressive
    assert kokoro.voices[0] == "af_heart"
    with pytest.raises(ConfigError, match="not a speech model"):
        FakeSpeech.like("gemma4-12b")
