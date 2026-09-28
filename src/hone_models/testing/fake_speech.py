"""`FakeSpeech`: an offline stand-in for `mk.speech(...)` that writes silent WAV files.

    tts = FakeSpeech()
    r = tts.synthesize("Hello there, and welcome.", out="hello.wav")   # 4 words -> 1.6 s of silence
    tts.calls                              # [(text, voice, speed, emotion, intensity), ...]
    FakeSpeech(expressive=True)            # also accepts the `chatterbox` voices (warm_male, ...)
    FakeSpeech.like("kokoro-82m")          # the voices, expressiveness and chunk size of a registry entry
    tts.loads, tts.frees                   # one each per call, or per `with tts.session():` block

Same splitting, span (`hone.models.speech`, model id `fake-speech`) and result as the real client; no GPU
lease, no model.
"""

from __future__ import annotations

from ..errors import ConfigError
from ..gpu import NullGpuLease
from ..ports import RecordSink
from ..providers.common import SpeechEngine, Style
from ..records import MemorySink
from ..registry import Capabilities, ModelConfig, Registry, load
from ..speech import SpeechClient

WORDS_PER_S = 2.5
KOKORO_VOICES = ("af_heart", "am_michael", "af_bella", "am_fenrir", "bf_emma", "bm_george")


class FakeSpeech(SpeechClient):
    """Silence of about `words / 2.5` seconds per chunk; records every call in `calls`. `expressive` sets
    the model's `expressive` capability (recorded on the span). The default voices are those of
    `kokoro-82m`, plus those of the `chatterbox` registry entry with `expressive=True` (Kokoro's stay
    first, for tests written before); `FakeSpeech.like(id)` copies a registry entry exactly.

    `sink` defaults to a `MemorySink` (read the spans from `tts.sink.spans`)."""

    def __init__(
        self,
        model_id: str = "fake-speech",
        *,
        voices: tuple[str, ...] | None = None,
        sample_rate: int = 24000,
        sink: RecordSink | None = None,
        expressive: bool = False,
    ) -> None:
        if voices is None:
            voices = KOKORO_VOICES
            if expressive:
                voices += tuple(load().get("chatterbox").capabilities.voices or ())
        caps = Capabilities(voices=list(voices), expressive=expressive)
        config = ModelConfig(id=model_id, provider="kokoro", kind="speech", capabilities=caps)
        super().__init__(config, sink if sink is not None else MemorySink(), lease=NullGpuLease())
        self.sample_rate = sample_rate
        self.calls: list[tuple[str, str, float, str | None, float | None]] = []
        self.loads = self.frees = 0  # model loads and frees, as the real client would do them

    @classmethod
    def like(
        cls, model_id: str, *, registry: Registry | None = None, sink: RecordSink | None = None
    ) -> FakeSpeech:
        """A fake with the id, voices, expressiveness and chunk size of the registry entry `model_id`
        (`kokoro-82m`, `chatterbox` or a project's own), so tests use the real model's names."""
        cfg = (registry or load()).get(model_id)
        if cfg.kind != "speech":
            raise ConfigError(f"model {cfg.id!r} is a {cfg.kind} model, not a speech model")
        caps = cfg.capabilities
        fake = cls(cfg.id, voices=tuple(caps.voices or ()), sink=sink, expressive=bool(caps.expressive))
        fake.config = cfg
        return fake

    def _ready(self, voice: str) -> None:
        pass  # nothing to install for silence

    def _open(self) -> SpeechEngine:
        self.loads += 1
        return _Silence(self)


class _Silence:
    """The fake's engine: records the call and returns silence."""

    def __init__(self, fake: FakeSpeech) -> None:
        self.fake = fake

    def __call__(self, chunks: list[str], voice: str, speed: float, style: Style) -> tuple[int, list[bytes]]:
        self.fake.calls.append((" ".join(chunks), voice, speed, *style))
        rate = self.fake.sample_rate
        frames = [round(len(c.split()) / WORDS_PER_S / speed * rate) for c in chunks]
        return rate, [b"\x00\x00" * max(n, 1) for n in frames]

    def close(self) -> None:
        self.fake.frees += 1
