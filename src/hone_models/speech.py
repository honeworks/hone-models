"""Text to speech: `mk.speech(...)` (design/current.md §2, changes 0006 and 0011).

tts = mk.speech("kokoro-82m")                     # extra `speech`; weights downloaded on first use
r = tts.synthesize("Hello there.", voice="am_michael", out="hello.wav")
r.duration_s, r.sample_rate, r.span_id
mk.speech("chatterbox").synthesize(text, emotion="encouraging", intensity=0.6, out="line.wav")

Text is split into paragraphs (blank lines); a paragraph longer than the model's `max_chunk_chars` is
split on sentences into chunks of about equal length. Chunks are synthesized one by one and joined with
`PAUSE_S` of silence, `PARAGRAPH_PAUSE_S` between paragraphs. Each call records one `hone.models.speech`
span and runs inside a GPU lease; the model is loaded and freed for every call, or once for all the calls
in a `with tts.session():` block (which holds the lease for its duration).
"""

from __future__ import annotations

import math
import re
import wave
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import gpu
from ._tracing import start_span
from .errors import ConfigError
from .ports import RecordSink
from .providers import SPEECH, SPEECH_READY, lookup, model_attributes
from .providers.common import SpeechEngine, Style
from .records import default_sink
from .registry import ModelConfig, Registry, load, require_client

MAX_CHUNK_CHARS = 400
PAUSE_S = 0.25
PARAGRAPH_PAUSE_S = 0.6
# The shared emotion vocabulary; expressive providers map each to their own controls (change 0011).
EMOTIONS = ("neutral", "calm", "warm", "serious", "curious", "encouraging", "playful", "excited")
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")
_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """Where one synthesized chunk (one or more whole sentences) sits in the WAV file, in seconds."""

    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class SpeechResult:
    """One synthesized WAV file (16-bit mono), the span that recorded it and where each chunk sits."""

    path: Path
    duration_s: float
    sample_rate: int
    voice: str
    model: str
    span_id: str
    segments: list[SpeechSegment] = field(default_factory=list[SpeechSegment])


def split_paragraphs(
    text: str, max_chars: int = MAX_CHUNK_CHARS, *, by_sentence: bool = False
) -> list[list[str]]:
    """The chunks of each paragraph (paragraphs are separated by blank lines). A paragraph that fits
    `max_chars` is one chunk; a longer one is split on sentences into chunks of about equal length, and
    an over-long sentence is cut at spaces. `by_sentence` makes every sentence its own chunk."""
    paragraphs = [" ".join(p.split()) for p in _PARAGRAPH.split(text)]
    if by_sentence:
        return [
            [c for sentence in _SENTENCE_END.split(p) for c in _cut(sentence, max_chars)]
            for p in paragraphs
            if p
        ]
    return [_pack(p, max_chars) for p in paragraphs if p]


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """All chunks of `split_paragraphs`, in order."""
    return [chunk for paragraph in split_paragraphs(text, max_chars) for chunk in paragraph]


def _pack(paragraph: str, max_chars: int) -> list[str]:
    target = math.ceil(len(paragraph) / math.ceil(len(paragraph) / max_chars))  # about equal chunks
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(paragraph):
        for piece in _cut(sentence, max_chars):
            joined = len(current) + 1 + len(piece)  # add the piece if that lands closer to the target
            if current and joined <= max_chars and joined - target < target - len(current):
                current = f"{current} {piece}"
            else:
                if current:
                    chunks.append(current)
                current = piece
    return [*chunks, current] if current else chunks


def _cut(sentence: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    while len(sentence) > max_chars:
        at = sentence.rfind(" ", 0, max_chars + 1)
        at = at if at > 0 else max_chars
        pieces.append(sentence[:at].strip())
        sentence = sentence[at:].strip()
    return [*pieces, sentence] if sentence else pieces


def join_pcm(chunks: list[bytes], sample_rate: int, pause_s: float = PAUSE_S) -> bytes:
    """16-bit mono PCM chunks joined with `pause_s` of silence between them."""
    silence = b"\x00\x00" * round(sample_rate * pause_s)
    return silence.join(chunks)


def join_paragraphs(chunks: list[bytes], sizes: list[int], sample_rate: int) -> bytes:
    """PCM chunks, `sizes[i]` of them for paragraph i, joined with `PAUSE_S` inside a paragraph and
    `PARAGRAPH_PAUSE_S` between paragraphs."""
    paragraphs: list[bytes] = []
    for size in sizes:
        paragraphs.append(join_pcm(chunks[:size], sample_rate))
        chunks = chunks[size:]
    return join_pcm(paragraphs, sample_rate, PARAGRAPH_PAUSE_S)


def segments(chunks: list[str], pcm: list[bytes], sizes: list[int], sample_rate: int) -> list[SpeechSegment]:
    """Where each chunk sits in the audio `join_paragraphs` makes of the same chunks."""
    pause, paragraph_pause = round(sample_rate * PAUSE_S), round(sample_rate * PARAGRAPH_PAUSE_S)
    found: list[SpeechSegment] = []
    at = i = 0  # in samples; chunk index
    for n, size in enumerate(sizes):
        at += paragraph_pause if n else 0
        for j in range(size):
            at += pause if j else 0
            start, at = at, at + len(pcm[i]) // 2
            found.append(SpeechSegment(chunks[i], round(start / sample_rate, 3), round(at / sample_rate, 3)))
            i += 1
    return found


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> float:
    """Write 16-bit mono PCM to `path` (parents created) and return its duration in seconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return len(pcm) / 2 / sample_rate


class SpeechClient:
    """Synthesizes speech with one model; records a `hone.models.speech` span per call."""

    def __init__(self, config: ModelConfig, sink: RecordSink, *, lease: Any = None) -> None:
        self.config = config
        self.sink = sink
        self.gpu = lease or gpu.GPU
        self._in_session = False
        self._engine: SpeechEngine | None = None

    @property
    def model_id(self) -> str:
        return self.config.id

    @property
    def voices(self) -> list[str]:
        """The registered voice names; the first is the default."""
        return list(self.config.capabilities.voices or [])

    @property
    def emotions(self) -> list[str]:
        """The accepted `emotion` values (`EMOTIONS`, shared by every model)."""
        return list(EMOTIONS)

    @property
    def expressive(self) -> bool:
        """Whether the model applies `emotion` and `intensity` (others accept and ignore them)."""
        return bool(self.config.capabilities.expressive)

    @property
    def max_chunk_chars(self) -> int:
        """The longest chunk synthesized at once (registry `defaults.max_chunk_chars`)."""
        return int(self.config.defaults.get("max_chunk_chars") or MAX_CHUNK_CHARS)

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        emotion: str | None = None,
        intensity: float | None = None,
        out: str | Path,
        trace: Mapping[str, str] | None = None,
        by_sentence: bool = False,
    ) -> SpeechResult:
        """Speak `text` into the WAV file `out` and return its path, duration, span id and `segments`
        (where each synthesized chunk starts and ends).

        `emotion` (one of `self.emotions`) and `intensity` (0.0 to 1.0) shape the delivery on expressive
        models (`self.expressive`); other models ignore them. Both are recorded on the span.
        `by_sentence=True` synthesizes each sentence on its own, so `segments` has one entry per sentence
        (sentence timings for captions and cuts, at the cost of prosody across sentences)."""
        voice = self._voice(voice)
        _check_style(speed, emotion, intensity)
        self._ready(voice)
        paragraphs = split_paragraphs(text, self.max_chunk_chars, by_sentence=by_sentence)
        chunks = [chunk for paragraph in paragraphs for chunk in paragraph]
        if not chunks:
            raise ConfigError("nothing to say: text is empty")
        attrs = {
            **model_attributes(self.config, "speech"),
            "hone.models.speech.input": text,
            "hone.models.speech.voice": voice,
            "hone.models.speech.speed": speed,
            "hone.models.speech.chars": len(text),
            "hone.models.speech.chunks": len(chunks),
            "hone.models.speech.paragraphs": len(paragraphs),
            "hone.models.speech.expressive": self.expressive,
            "hone.models.speech.session": self._in_session,
        }
        if emotion is not None:
            attrs["hone.models.speech.emotion"] = emotion
        if intensity is not None:
            attrs["hone.models.speech.intensity"] = intensity
        path = Path(out)
        with start_span("hone.models.speech", self.sink, attrs, trace=trace) as span:
            lease = nullcontext() if self._in_session else self._lease(trace)  # a session holds its own
            with lease:
                span["attributes"]["hone.models.speech.loaded"] = self._engine is None
                rate, pcm = self._render(chunks, voice, speed, (emotion, intensity))
            sizes = [len(p) for p in paragraphs]
            duration = write_wav(path, join_paragraphs(pcm, sizes, rate), rate)
            span["attributes"].update(
                {"hone.models.speech.duration_s": round(duration, 3), "hone.models.speech.sample_rate": rate}
            )
        timings = segments(chunks, pcm, sizes, rate)
        return SpeechResult(path, duration, rate, voice, self.config.id, span["span_id"], timings)

    def _ready(self, voice: str) -> None:
        """Raise `ConfigError` before the GPU lease when the model cannot speak with `voice` here."""
        check = SPEECH_READY.get(self.config.provider)
        if check is not None:
            check(voice)

    def _voice(self, voice: str | None) -> str:
        voices = self.voices
        if voice is None:
            if not voices:
                raise ConfigError(f"model {self.model_id!r} lists no voices; pass voice=")
            return voices[0]
        if voices and voice not in voices:
            raise ConfigError(f"unknown voice {voice!r} for {self.model_id!r}; choose one of {voices}")
        return voice

    @contextmanager
    def session(self, *, trace: Mapping[str, str] | None = None) -> Generator[SpeechClient]:
        """Keep the model loaded and the GPU lease held for the `synthesize` calls in the block, and free
        both once at the end. Without a session each call loads and frees the model itself.

            with mk.speech("chatterbox").session() as tts:
                for i, line in enumerate(lines):
                    tts.synthesize(line, emotion="warm", out=f"line_{i}.wav")
        """
        if self._in_session:
            raise ConfigError(f"a session of {self.model_id!r} is already open")
        with self._lease(trace):
            self._in_session = True
            try:
                yield self
            finally:
                self._in_session = False
                engine, self._engine = self._engine, None
                if engine is not None:
                    engine.close()

    def _lease(self, trace: Mapping[str, str] | None) -> AbstractContextManager[Any]:
        return self.gpu.lease(self.config.id, self.config.capabilities.vram_gb or 1.0, trace=trace)

    def _render(self, chunks: list[str], voice: str, speed: float, style: Style) -> tuple[int, list[bytes]]:
        if self._in_session:  # load on the first call, keep until the session ends
            self._engine = self._engine or self._open()
            return self._engine(chunks, voice, speed, style)
        engine = self._open()
        try:
            return engine(chunks, voice, speed, style)
        finally:
            engine.close()

    def _open(self) -> SpeechEngine:
        return lookup(SPEECH, self.config, "speech")(self.config)


def _check_style(speed: float, emotion: str | None, intensity: float | None) -> None:
    if not 0.5 <= speed <= 2.0:
        raise ConfigError(f"speed must be between 0.5 and 2.0, got {speed}")
    if emotion is not None and emotion not in EMOTIONS:
        raise ConfigError(f"unknown emotion {emotion!r}; choose one of {list(EMOTIONS)}")
    if intensity is not None and not 0.0 <= intensity <= 1.0:
        raise ConfigError(f"intensity must be between 0.0 and 1.0, got {intensity}")


def speech(
    model_id: str, *, registry: Registry | None = None, sink: RecordSink | None = None
) -> SpeechClient:
    """A `SpeechClient` for a registered speech model (`kind = "speech"`): `kokoro-82m`, `chatterbox`."""
    cfg = (registry or load()).get(model_id)
    require_client(cfg)
    if cfg.kind != "speech":
        raise ConfigError(f"model {cfg.id!r} is a {cfg.kind} model, not a speech model")
    return SpeechClient(cfg, sink or default_sink())
