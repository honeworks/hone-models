"""The Chatterbox provider with a stand-in for the `chatterbox` package (the real model: tests/gpu)."""

import types
from pathlib import Path
from typing import Any

import pytest

import hone_models as mk
from hone_models.errors import ConfigError, ProviderError
from hone_models.providers import chatterbox
from hone_models.speech import EMOTIONS

torch: Any = pytest.importorskip("torch")  # installed with the speech extras


class StandIn:
    """Records what the provider asks of ChatterboxTTS; returns 0.1 s of a constant signal."""

    made: list["StandIn"] = []  # noqa: RUF012

    def __init__(self, device: str) -> None:
        self.device, self.sr, self.refs, self.generated = device, 24000, [], []
        self.conds = "bundled"
        StandIn.made.append(self)

    @classmethod
    def from_pretrained(cls, device: str) -> "StandIn":
        return cls(device)

    def prepare_conditionals(self, path: str, exaggeration: float) -> None:
        assert Path(path).is_file()
        self.refs.append((Path(path).name, exaggeration))
        self.conds = Path(path).stem

    def generate(self, text: str, exaggeration: float, cfg_weight: float) -> Any:
        self.generated.append((text, exaggeration, cfg_weight, self.conds))
        return torch.full((1, 2400), 0.5)


@pytest.fixture
def standin(monkeypatch: pytest.MonkeyPatch) -> type[StandIn]:
    StandIn.made = []
    real = chatterbox._import
    fake_tts = types.SimpleNamespace(ChatterboxTTS=StandIn)
    monkeypatch.setattr(
        chatterbox, "_import", lambda name: fake_tts if name == "chatterbox.tts" else real(name)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # never touch the shared GPU here
    return StandIn


def cfg() -> mk.registry.ModelConfig:
    return mk.registry.load().get("chatterbox")


def synthesize(
    chunks: list[str], voice: str, speed: float, style: tuple[Any, Any]
) -> tuple[int, list[bytes]]:
    engine = chatterbox.Engine(cfg())
    try:
        return engine(chunks, voice, speed, style)
    finally:
        engine.close()


def test_every_shared_emotion_has_a_chatterbox_style() -> None:
    assert set(chatterbox.STYLES) == set(EMOTIONS)


def test_controls_map_intensity_to_exaggeration_and_emotion_to_pacing() -> None:
    assert chatterbox.controls(None, None) == (0.5, 0.5)  # Chatterbox's own defaults
    assert chatterbox.controls("excited", None) == (1.0, 0.3)
    assert chatterbox.controls("calm", None) == (0.35, 0.5)
    assert chatterbox.controls("calm", 1.0) == (1.25, 0.5)  # intensity overrides the emotion's default
    assert chatterbox.controls(None, 0.0) == (0.25, 0.5)
    livelier = [chatterbox.controls(e, None) for e in ("calm", "neutral", "warm", "encouraging", "excited")]
    assert [x for x, _ in livelier] == sorted(x for x, _ in livelier)
    assert [c for _, c in livelier] == sorted((c for _, c in livelier), reverse=True)


def test_synthesize_uses_the_voice_clip_and_emotion(standin: type[StandIn]) -> None:
    rate, pcm = synthesize(["One.", "Two."], "warm_male", 1.0, ("excited", None))
    (model,) = standin.made
    assert rate == 24000
    assert model.device == "cpu"
    assert model.refs == [("warm_male.flac", 1.0)]
    assert model.generated == [("One.", 1.0, 0.3, "warm_male"), ("Two.", 1.0, 0.3, "warm_male")]
    assert [len(p) for p in pcm] == [4800, 4800]  # 2400 samples of 16-bit PCM each
    assert pcm[0][:2] == (16383).to_bytes(2, "little", signed=True)


def test_default_voice_needs_no_clip_and_speed_time_stretches(standin: type[StandIn]) -> None:
    _, pcm = synthesize(["One."], "chatterbox_default", 2.0, (None, 0.5))
    (model,) = standin.made
    assert model.refs == []
    assert model.generated == [("One.", 0.75, 0.5, "bundled")]
    assert abs(len(pcm[0]) // 2 - 1200) <= 2  # twice as fast: half the samples


def test_packaged_voice_clips_exist() -> None:
    folder = Path(chatterbox.__file__).parents[1] / "data" / "voices"
    for voice in cfg().capabilities.voices or []:
        assert voice == chatterbox.DEFAULT_VOICE or (folder / f"{voice}.flac").is_file()


def test_unknown_voice_clip_and_library_failures_are_reported(
    standin: type[StandIn], monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ConfigError, match="no voice 'nobody'"):
        synthesize(["One."], "nobody", 1.0, (None, None))

    def broken(*_: object, **__: object) -> None:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(StandIn, "generate", broken)
    with pytest.raises(ProviderError, match="Chatterbox synthesis failed: RuntimeError: CUDA out of memory"):
        synthesize(["One."], "warm_female", 1.0, (None, None))
    monkeypatch.setattr(StandIn, "generate", lambda *_, **__: torch.zeros((1, 0)))
    with pytest.raises(ProviderError, match="no audio"):
        synthesize(["One."], "warm_female", 1.0, (None, None))


def test_chatterbox_without_the_extra_names_it(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> None:
        raise ImportError(name)

    config = cfg()
    monkeypatch.setattr(chatterbox.importlib, "import_module", missing)
    with pytest.raises(ConfigError, match=r"hone-models\[expressive\]"):
        chatterbox.Engine(config)


def test_client_passes_emotion_to_chatterbox(tmp_path: Path, standin: type[StandIn]) -> None:
    sink = mk.records.MemorySink()
    tts = mk.speech("chatterbox", sink=sink)
    tts.gpu = mk.gpu.NullGpuLease()
    r = tts.synthesize("Great work.\n\nNow the next step.", emotion="encouraging", out=tmp_path / "x.wav")
    (model,) = standin.made
    assert [g[:3] for g in model.generated] == [("Great work.", 0.75, 0.4), ("Now the next step.", 0.75, 0.4)]
    assert r.duration_s == pytest.approx(0.1 + 0.6 + 0.1)
    assert sink.spans[0]["attributes"]["hone.models.speech.expressive"] is True


def test_engine_prepares_a_voice_only_when_it_changes(standin: type[StandIn]) -> None:
    engine = chatterbox.Engine(cfg())
    for voice in ("warm_female", "warm_female", "warm_male", "chatterbox_default", "warm_male"):
        engine(["Hi."], voice, 1.0, ("warm", None))
    engine.close()
    (model,) = standin.made
    assert [name for name, _ in model.refs] == ["warm_female.flac", "warm_male.flac", "warm_male.flac"]
    assert [g[3] for g in model.generated] == [
        "warm_female",
        "warm_female",
        "warm_male",
        "bundled",
        "warm_male",
    ]
    assert engine.model is None  # freed


def test_session_loads_chatterbox_once(tmp_path: Path, standin: type[StandIn]) -> None:
    tts = mk.speech("chatterbox", sink=mk.records.NullSink())
    tts.gpu = mk.gpu.NullGpuLease()
    with tts.session():
        for i, emotion in enumerate(("curious", "serious", "encouraging")):
            tts.synthesize("One line.", emotion=emotion, out=tmp_path / f"{i}.wav")
    assert len(standin.made) == 1
    assert [g[1:3] for g in standin.made[0].generated] == [(0.7, 0.45), (0.4, 0.55), (0.75, 0.4)]
