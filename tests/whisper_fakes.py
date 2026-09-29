"""Fake `faster_whisper` and `ctranslate2` modules, as the speech tests fake `kokoro`: installed into
`sys.modules`, so the real provider (`providers/faster_whisper.py`) runs end to end without a model."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import hone_models as mk

WHISPER_FIXTURES = Path(__file__).parent / "fixtures" / "whisper"
# What the fake model hears: faster-whisper's own shapes (leading spaces on texts and words).
LINES = [(" Hello there, and welcome.", 0.0, 1.6), (" This is the second line.", 1.9, 3.7)]


def registry() -> mk.registry.Registry:
    """The packaged registry plus `test-whisper` (CUDA) and `test-whisper-cpu` (tests/fixtures/whisper)."""
    return mk.registry.load([WHISPER_FIXTURES / "models.toml"])


@dataclass
class FakeWhisper:
    """The state behind the fake modules: what was loaded, asked and unloaded."""

    cuda_devices: int = 0
    loads: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    calls: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    unloads: int = 0
    segment_delay_s: float = 0.0
    fail_on_load: Exception | None = None
    lines: list[tuple[str, float, float]] = field(default_factory=lambda: list(LINES))

    def modules(self) -> tuple[ModuleType, ModuleType]:
        whisper = ModuleType("faster_whisper")
        whisper.WhisperModel = self._model  # type: ignore[attr-defined]
        ct2 = ModuleType("ctranslate2")
        ct2.get_cuda_device_count = lambda: self.cuda_devices  # type: ignore[attr-defined]
        return whisper, ct2

    def _model(self, name: str, **options: Any) -> Any:
        if self.fail_on_load is not None:
            raise self.fail_on_load
        self.loads.append({"name": name, **options})
        state = self

        class Inner:
            def unload_model(self) -> None:
                state.unloads += 1

        return SimpleNamespace(model=Inner(), transcribe=self._transcribe)

    def _transcribe(self, audio: str, **options: Any) -> tuple[Any, Any]:
        self.calls.append({"audio": audio, **options})
        info = SimpleNamespace(language=options.get("language") or "en", duration=4.0)
        return self._segments(options["word_timestamps"]), info

    def _segments(self, words: bool) -> Any:
        for text, start, end in self.lines:
            time.sleep(self.segment_delay_s)
            parts = text.split()
            step = (end - start) / len(parts)
            heard = [
                SimpleNamespace(
                    word=f" {w}", start=start + i * step, end=start + (i + 1) * step, probability=0.9
                )
                for i, w in enumerate(parts)
            ]
            yield SimpleNamespace(text=text, start=start, end=end, words=heard if words else None)


def install(monkeypatch: pytest.MonkeyPatch, **state: Any) -> FakeWhisper:
    """Put fake `faster_whisper` and `ctranslate2` modules in `sys.modules` and return their state."""
    fake = FakeWhisper(**state)
    whisper, ct2 = fake.modules()
    monkeypatch.setitem(sys.modules, "faster_whisper", whisper)
    monkeypatch.setitem(sys.modules, "ctranslate2", ct2)
    return fake
