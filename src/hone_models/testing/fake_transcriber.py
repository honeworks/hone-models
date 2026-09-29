"""`FakeTranscriber`: an offline stand-in for `mk.transcriber(...)` that "hears" a scripted text.

    stt = FakeTranscriber(text="Hello there, and welcome.")
    t = stt.transcribe("take.wav")          # 4 words, 0.4 s each: Word("Hello", 0.0, 0.4, 1.0), ...
    stt.text = "Something else."            # what the next calls hear
    stt.calls                               # [(audio path, language, prompt, words), ...]
    FakeTranscriber.like("faster-whisper-large-v3-turbo")   # the id and capabilities of an entry
    stt.loads, stt.frees                    # one each per call, or per `with stt.session():` block

Same checks, span (`hone.models.transcribe`, model id `fake-transcriber`) and result as the real client;
no GPU lease, no model. The audio file must exist; its content is never read.
"""

from __future__ import annotations

from pathlib import Path

from .._transcript import Heard, TranscribeEngine, TranscriptSegment, Word
from ..errors import ConfigError
from ..gpu import NullGpuLease
from ..ports import RecordSink
from ..records import MemorySink
from ..registry import Capabilities, ModelConfig, Registry, load
from ..transcribe import Transcriber

WORD_S = 0.4  # every word lasts 0.4 s (2.5 words per second, as FakeSpeech speaks)


class FakeTranscriber(Transcriber):
    """Hears `text` in every file: one segment per sentence line, words `WORD_S` apart; records every
    call in `calls`. `language` is the one asked for, else `language` given here.

    `sink` defaults to a `MemorySink` (read the spans from `stt.sink.spans`)."""

    def __init__(
        self,
        model_id: str = "fake-transcriber",
        *,
        text: str = "Hello there, and welcome.",
        language: str = "en",
        sink: RecordSink | None = None,
    ) -> None:
        caps = Capabilities(word_timestamps=True)
        config = ModelConfig(id=model_id, provider="faster_whisper", kind="transcription", capabilities=caps)
        super().__init__(config, sink if sink is not None else MemorySink(), lease=NullGpuLease())
        self.text = text
        self.language = language
        self.calls: list[tuple[Path, str | None, str | None, bool]] = []
        self.loads = self.frees = 0  # model loads and frees, as the real client would do them

    @classmethod
    def like(
        cls, model_id: str, *, registry: Registry | None = None, sink: RecordSink | None = None
    ) -> FakeTranscriber:
        """A fake with the id and capabilities of the registry entry `model_id`."""
        cfg = (registry or load()).get(model_id)
        if cfg.kind != "transcription":
            raise ConfigError(
                f"model {cfg.id!r} is a {cfg.kind} model; mk.transcriber() takes transcription models"
            )
        fake = cls(cfg.id, sink=sink)
        fake.config = cfg
        return fake

    def _ready(self) -> None:
        pass  # nothing to install for a scripted text

    def _open(self) -> TranscribeEngine:
        self.loads += 1
        return _Scripted(self)


class _Scripted:
    """The fake's engine: records the call and returns the scripted text with evenly spaced words."""

    def __init__(self, fake: FakeTranscriber) -> None:
        self.fake = fake

    def __call__(
        self, audio: Path, language: str | None, prompt: str | None, words: bool, deadline: float
    ) -> Heard:
        del deadline
        self.fake.calls.append((audio, language, prompt, words))
        segments: list[TranscriptSegment] = []
        at = 0.0
        for line in (line.strip() for line in self.fake.text.splitlines()):
            if not line:
                continue
            heard = [Word(w, round(at + i * WORD_S, 3), round(at + (i + 1) * WORD_S, 3), 1.0) for i, w in
                     enumerate(line.split())]  # fmt: skip
            end = round(at + len(heard) * WORD_S, 3)
            segments.append(TranscriptSegment(line, at, end, heard if words else []))
            at = end
        return Heard(language or self.fake.language, at, segments)

    def close(self) -> None:
        self.fake.frees += 1
