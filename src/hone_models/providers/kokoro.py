"""Kokoro-82M text to speech (Apache-2.0), extra `speech`. Imported lazily; the core never needs it.

The weights and voices come from Hugging Face (`hexgrad/Kokoro-82M`) on first use into its cache
(`HF_HOME`, default `~/.cache/huggingface`). `Engine` holds the loaded model: the speech client opens
one per call (or once per `session()`) and closes it afterwards (CUDA cache emptied), so it never holds
memory on the shared GPU between calls.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from ..errors import ConfigError, ProviderError
from ..registry import ModelConfig
from .common import Style, free_torch, library_errors

SAMPLE_RATE = 24000
REPO_ID = "hexgrad/Kokoro-82M"
SPACY_WHEEL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)


def check_ready(voice: str) -> None:
    """Fail before any GPU work when an English voice cannot run: Kokoro's English phonemizer (misaki)
    needs spaCy's `en_core_web_sm`, which PyPI cannot carry as a dependency and which misaki would
    otherwise try to pip-install at run time (change 0008). Without the extra there is nothing to check
    here; `Engine` names the extra."""
    english = voice[:1] in ("a", "b")
    if english and importlib.util.find_spec("misaki") and not importlib.util.find_spec("en_core_web_sm"):
        raise ConfigError(
            "Kokoro's English voices need spaCy's model en_core_web_sm, which is not installed: "
            f"`uv pip install {SPACY_WHEEL}`, or add `en-core-web-sm @ {SPACY_WHEEL}` to your dependencies "
            "(see docs/speech.md)"
        )


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ConfigError(
            "Kokoro speech needs the `speech` extra (Python < 3.13): pip install 'hone-models[speech]'"
        ) from exc


class Engine:
    """Kokoro loaded once (`KModel` plus a `KPipeline` per language, made on first use); `close()` frees
    it. The voice's first letter picks the language (`a` American English, `b` British English).
    Emotion and intensity are ignored: Kokoro has no expression control."""

    def __init__(self, cfg: ModelConfig) -> None:
        self.torch = _import("torch")
        self.kokoro = _import("kokoro")
        cuda = self.torch.cuda.is_available()
        self.device = str(cfg.defaults.get("device") or ("cuda" if cuda else "cpu"))
        self.repo = cfg.model or REPO_ID
        self.pipelines: dict[str, Any] = {}
        with library_errors("Kokoro synthesis"):
            self.model: Any = self.kokoro.KModel(repo_id=self.repo).to(self.device).eval()

    def __call__(self, chunks: list[str], voice: str, speed: float, style: Style) -> tuple[int, list[bytes]]:
        del style
        with library_errors("Kokoro synthesis"):
            if voice[0] not in self.pipelines:
                self.pipelines[voice[0]] = self.kokoro.KPipeline(
                    lang_code=voice[0], repo_id=self.repo, model=self.model
                )
            pipeline = self.pipelines[voice[0]]
            return SAMPLE_RATE, [_render(pipeline, self.torch, chunk, voice, speed) for chunk in chunks]

    def close(self) -> None:
        self.pipelines.clear()
        self.model = None
        free_torch(self.torch)


def _render(pipeline: Any, torch: Any, chunk: str, voice: str, speed: float) -> bytes:
    parts = [
        r.audio for r in pipeline(chunk, voice=voice, speed=speed, split_pattern=None) if r.audio is not None
    ]
    if not parts:
        raise ProviderError(f"Kokoro produced no audio for {chunk[:60]!r}")
    audio = torch.cat(parts).clamp(-1.0, 1.0)
    return (audio * 32767).to(torch.int16).cpu().numpy().tobytes()
