"""Transcription with word timestamps: `mk.transcriber(...)` (change 0015 §1).

    stt = mk.transcriber("faster-whisper-large-v3-turbo")      # extra `transcribe`
    t = stt.transcribe("takes/take_1.flac", language="en", prompt=lyrics)   # prompt biases the words
    t.text, t.language, t.duration_s, t.segments[0].words[0]   # Word(text, start_s, end_s, probability)
    with stt.session():                                         # lease and model held for the block
        for take in takes:
            stt.transcribe(take)

Each call records one `hone.models.transcribe` span and runs inside a GPU lease; the model is loaded and
freed for every call, or once for all the calls in a `with stt.session():` block (which holds the lease
for its duration), as speech does. Aligning known lyrics to the heard words stays with the caller: it is
not a model call.
"""

from __future__ import annotations

import time
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import gpu
from ._media_files import file_record
from ._tracing import start_span
from ._transcript import Heard, TranscribeEngine
from ._transcript import TranscriptSegment as TranscriptSegment  # noqa: PLC0414 - part of this API
from ._transcript import Word as Word  # noqa: PLC0414 - part of this module's API
from .errors import CapabilityError, ConfigError
from .ports import RecordSink
from .providers import TRANSCRIBE, TRANSCRIBE_READY, lookup, model_attributes
from .records import default_sink
from .registry import ModelConfig, Registry, load


@dataclass(frozen=True, slots=True)
class Transcript:
    """What was heard: the whole `text`, its `language`, the audio's `duration_s` (`None` when unknown),
    the `segments` in order, every word of them in `words` (empty with `words=False`), the model and the
    span that recorded the call."""

    text: str
    language: str | None
    duration_s: float | None
    segments: list[TranscriptSegment]
    model: str
    span_id: str
    words: list[Word] = field(default_factory=list[Word])


class Transcriber:
    """Transcribes audio files with one model; records a `hone.models.transcribe` span per call."""

    def __init__(self, config: ModelConfig, sink: RecordSink, *, lease: Any = None) -> None:
        self.config = config
        self.sink = sink
        self.gpu = lease or gpu.GPU
        self._in_session = False
        self._engine: TranscribeEngine | None = None

    @property
    def model_id(self) -> str:
        return self.config.id

    def transcribe(
        self,
        audio: str | Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
        words: bool = True,
        timeout_s: float | None = None,
        trace: Mapping[str, str] | None = None,
    ) -> Transcript:
        """Transcribe the audio file `audio` (any format the model reads: WAV, FLAC, MP3, ...).

        `language` is an ISO code (`"en"`); `None` lets the model detect it. `prompt` is text the model
        is biased toward (known lyrics, names). `words=False` skips word timestamps. `timeout_s`
        (default: the entry's `max_timeout_s`) stops decoding and raises `ModelTimeout` when exceeded."""
        path = self._checked(audio, words, timeout_s)
        self._ready()
        attrs = {
            **model_attributes(self.config, "transcription"),
            "hone.models.transcribe.audio": file_record(path),
            "hone.models.transcribe.language": language,
            "hone.models.transcribe.session": self._in_session,
        }
        if prompt is not None:
            attrs["hone.models.transcribe.prompt"] = prompt
        deadline = time.monotonic() + (timeout_s or self.config.max_timeout_s)
        with start_span("hone.models.transcribe", self.sink, attrs, trace=trace) as span:
            lease = nullcontext() if self._in_session else self._lease(trace)  # a session holds its own
            with lease:
                span["attributes"]["hone.models.transcribe.loaded"] = self._engine is None
                heard = self._run(path, (language, prompt, words), deadline)
            result = _transcript(heard, self.config.id, span["span_id"])
            span["attributes"].update(_recorded(result))
        return result

    @contextmanager
    def session(self, *, trace: Mapping[str, str] | None = None) -> Generator[Transcriber]:
        """Keep the model loaded and the GPU lease held for the `transcribe` calls in the block, and free
        both once at the end (also on an exception). Sessions of one transcriber do not nest.

            with mk.transcriber("faster-whisper-large-v3-turbo").session() as stt:
                heard = [stt.transcribe(take) for take in takes]
        """
        if self._in_session:
            raise ConfigError(f"a session of {self.model_id!r} is already open")
        self._ready()
        with self._lease(trace):
            self._in_session = True
            try:
                yield self
            finally:
                self._in_session = False
                engine, self._engine = self._engine, None
                if engine is not None:
                    engine.close()

    def _checked(self, audio: str | Path, words: bool, timeout_s: float | None) -> Path:
        path = Path(audio)
        if not path.is_file():
            raise ConfigError(f"audio file {path} does not exist")
        if words and self.config.capabilities.word_timestamps is False:
            raise CapabilityError(f"model {self.model_id!r} gives no word timestamps; pass words=False")
        if timeout_s is not None and timeout_s <= 0:
            raise ConfigError(f"timeout_s must be positive, got {timeout_s}")
        return path

    def _ready(self) -> None:
        """Raise `ConfigError` before the GPU lease when the model cannot run here (change 0015 §2)."""
        check = TRANSCRIBE_READY.get(self.config.provider)
        if check is not None:
            check(self.config)

    def _lease(self, trace: Mapping[str, str] | None) -> AbstractContextManager[Any]:
        return self.gpu.lease(self.config.id, self.config.capabilities.vram_gb or 1.0, trace=trace)

    def _run(self, path: Path, ask: tuple[str | None, str | None, bool], deadline: float) -> Heard:
        if self._in_session:  # load on the first call, keep until the session ends
            self._engine = self._engine or self._open()
            return self._engine(path, *ask, deadline)
        engine = self._open()
        try:
            return engine(path, *ask, deadline)
        finally:
            engine.close()

    def _open(self) -> TranscribeEngine:
        return lookup(TRANSCRIBE, self.config, "transcription")(self.config)


def _transcript(heard: Heard, model: str, span_id: str) -> Transcript:
    text = " ".join(s.text for s in heard.segments if s.text)
    words = [w for s in heard.segments for w in s.words]
    return Transcript(text, heard.language, heard.duration_s, heard.segments, model, span_id, words)


def _recorded(t: Transcript) -> dict[str, Any]:
    """The span attributes known after the call; `text` and `words` are content."""
    return {
        "hone.models.transcribe.language": t.language,
        "hone.models.transcribe.duration_s": t.duration_s,
        "hone.models.transcribe.words_count": len(t.words),
        "hone.models.transcribe.text": t.text,
        "hone.models.transcribe.words": [w.record() for w in t.words],
    }


def transcriber(
    model_id: str, *, registry: Registry | None = None, sink: RecordSink | None = None
) -> Transcriber:
    """A `Transcriber` for a registered transcription model (`kind = "transcription"`)."""
    cfg = (registry or load()).get(model_id)
    if cfg.kind != "transcription":
        raise ConfigError(
            f"model {cfg.id!r} is a {cfg.kind} model; mk.transcriber() takes transcription models"
        )
    return Transcriber(cfg, sink or default_sink())
