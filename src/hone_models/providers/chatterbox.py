"""Chatterbox text to speech (Resemble AI, MIT), extra `expressive`. Imported lazily (change 0011).

The weights (about 3 GB) come from Hugging Face (`ResembleAI/chatterbox`) on first use into its cache
(`HF_HOME`). A voice is a reference clip: the packaged ones in `hone_models/data/voices/` were rendered by
Kokoro's synthetic voices (`scripts/make-expressive-voices.py`); `chatterbox_default` is the voice that
ships with the weights. `Engine` holds the loaded model; the speech client opens one per call (or
once per `session()`) and closes it afterwards, like Kokoro.

Emotion and intensity map to Chatterbox's two controls: `exaggeration = 0.25 + intensity` (0.25 to 1.25;
Chatterbox's own default 0.5 is intensity 0.25) and `cfg_weight`, lower for livelier emotions so the
faster delivery that exaggeration brings is slowed down again.
"""

from __future__ import annotations

import importlib
from importlib import resources
from typing import Any

from ..errors import ConfigError, ProviderError
from ..registry import ModelConfig
from .common import Style, free_torch, library_errors

DEFAULT_VOICE = "chatterbox_default"
# emotion -> (default intensity, cfg_weight); the keys are hone_models.speech.EMOTIONS
STYLES: dict[str, tuple[float, float]] = {
    "neutral": (0.25, 0.5),
    "calm": (0.1, 0.5),
    "warm": (0.35, 0.5),
    "serious": (0.15, 0.55),
    "curious": (0.45, 0.45),
    "encouraging": (0.5, 0.4),
    "playful": (0.6, 0.4),
    "excited": (0.75, 0.3),
}


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ConfigError(
            "Chatterbox speech needs the `expressive` extra (Python < 3.13): "
            "pip install 'hone-models[expressive]' (see docs/speech.md for the torch pins)"
        ) from exc


def controls(emotion: str | None, intensity: float | None) -> tuple[float, float]:
    """Chatterbox's `(exaggeration, cfg_weight)` for an emotion and intensity (`None`: neutral / its
    default)."""
    default, cfg_weight = STYLES[emotion or "neutral"]
    return round(0.25 + (default if intensity is None else intensity), 3), cfg_weight


class Engine:
    """Chatterbox loaded once; a voice's reference clip is prepared when the voice changes. `close()`
    frees it. Each call is seeded (`defaults.seed`, default 0), so a text sounds the same in or out of a
    session."""

    def __init__(self, cfg: ModelConfig) -> None:
        self.torch = _import("torch")
        tts = _import("chatterbox.tts")
        self.seed = int(cfg.defaults.get("seed", 0))
        cuda = self.torch.cuda.is_available()
        device = str(cfg.defaults.get("device") or ("cuda" if cuda else "cpu"))
        self.voice = DEFAULT_VOICE  # the voice bundled with the weights is prepared on load
        with library_errors("Chatterbox synthesis"):
            self.model: Any = tts.ChatterboxTTS.from_pretrained(device)
        self.bundled = self.model.conds  # to switch back to the bundled voice

    def __call__(self, chunks: list[str], voice: str, speed: float, style: Style) -> tuple[int, list[bytes]]:
        exaggeration, cfg_weight = controls(*style)
        with library_errors("Chatterbox synthesis"):
            if voice == DEFAULT_VOICE:
                self.model.conds = self.bundled
            elif voice != self.voice:
                with resources.as_file(_voice_clip(voice)) as ref:
                    self.model.prepare_conditionals(str(ref), exaggeration=exaggeration)
            self.voice = voice
            self.torch.manual_seed(self.seed)
            knobs = {"exaggeration": exaggeration, "cfg_weight": cfg_weight}
            return self.model.sr, [_render(self.model, self.torch, chunk, knobs, speed) for chunk in chunks]

    def close(self) -> None:
        self.model = self.bundled = None
        free_torch(self.torch)


def _voice_clip(voice: str) -> Any:
    folder = resources.files("hone_models") / "data" / "voices"
    clip = folder / f"{voice}.flac"
    if not clip.is_file():
        raise ConfigError(f"Chatterbox has no voice {voice!r}; the packaged voices are in {folder}")
    return clip


def _render(model: Any, torch: Any, chunk: str, knobs: dict[str, float], speed: float) -> bytes:
    wav = model.generate(chunk, **knobs).squeeze(0)
    if speed != 1.0:
        librosa = _import("librosa")
        wav = torch.from_numpy(librosa.effects.time_stretch(wav.numpy(), rate=speed))
    if wav.numel() == 0:
        raise ProviderError(f"Chatterbox produced no audio for {chunk[:60]!r}")
    return (wav.clamp(-1.0, 1.0) * 32767).to(torch.int16).cpu().numpy().tobytes()
