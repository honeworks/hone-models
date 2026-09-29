"""What a transcription engine hands back (change 0015 §1): words and segments with their times.

A leaf module, so the providers can build these without importing the client (`transcribe.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Word:
    """One heard word, its start and end in seconds and the model's probability for it (0 to 1)."""

    text: str
    start_s: float
    end_s: float
    probability: float | None = None

    def record(self) -> dict[str, Any]:
        """The span form: `{"text", "start_s", "end_s", "probability"}`."""
        return {
            "text": self.text,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "probability": self.probability,
        }


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One stretch of speech: its text, start and end in seconds, and its words (empty when the call
    asked for no word timestamps)."""

    text: str
    start_s: float
    end_s: float
    words: list[Word] = field(default_factory=list[Word])


@dataclass(frozen=True, slots=True)
class Heard:
    """An engine's answer: the language it heard (or was told), the audio's duration in seconds (`None`
    when unknown) and the segments in order."""

    language: str | None
    duration_s: float | None
    segments: list[TranscriptSegment]


class TranscribeEngine(Protocol):
    """A loaded transcription model: call it for many files, then `close()` it to free its memory."""

    def __call__(
        self, audio: Path, language: str | None, prompt: str | None, words: bool, deadline: float
    ) -> Heard:
        """Transcribe `audio`; raise `ModelTimeout` once `time.monotonic()` passes `deadline`."""
        ...

    def close(self) -> None:
        """Free the model (and its GPU memory)."""
        ...
