"""Provider table: registry `provider` name -> functions. Each provider is a plain module."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from .._transcript import TranscribeEngine
from ..errors import ConfigError
from ..registry import Capabilities, ModelConfig
from . import chatterbox, comfyui, faster_whisper, kokoro, litellm, ollama, openai_compat, openai_media
from .common import ChatReply, ChatRequest, MediaProvider, SpeechEngine

CHAT: dict[str, Callable[[ModelConfig, ChatRequest], ChatReply]] = {
    "ollama": ollama.chat,
    "openai_compatible": openai_compat.chat,
    "litellm": litellm.chat,
}
EMBED: dict[str, Callable[[ModelConfig, list[str], float], list[list[float]]]] = {
    "ollama": ollama.embed,
    "openai_compatible": openai_compat.embed,
    "litellm": litellm.embed,
}
# Speech: config -> a loaded engine; engine(chunks, voice, speed, (emotion, intensity)) -> (rate, PCM per
# chunk); engine.close() frees it.
SPEECH: dict[str, Callable[[ModelConfig], SpeechEngine]] = {
    "kokoro": kokoro.Engine,
    "chatterbox": chatterbox.Engine,
}
# Speech: checks that fail before the GPU lease when a voice cannot run here (change 0008).
SPEECH_READY: dict[str, Callable[[str], None]] = {"kokoro": kokoro.check_ready}
# Images, music and video (change 0015): which inputs an entry takes, one job, a session (common.py).
MEDIA: dict[str, MediaProvider] = {
    "comfyui": comfyui.PROVIDER,
    "openai_compatible": openai_media.PROVIDER,
}
# Transcription (change 0015): config -> a loaded engine; engine(audio, language, prompt, words, deadline)
# -> Heard; engine.close() frees it. The checks fail before the GPU lease when the model cannot run here.
TRANSCRIBE: dict[str, Callable[[ModelConfig], TranscribeEngine]] = {"faster_whisper": faster_whisper.Engine}
TRANSCRIBE_READY: dict[str, Callable[[ModelConfig], None]] = {"faster_whisper": faster_whisper.check_ready}
# Capability probes for ad-hoc ids.
PROBE: dict[str, Callable[[ModelConfig], Capabilities]] = {"ollama": ollama.probe, "litellm": litellm.probe}
# `gen_ai.provider.name` (OTel GenAI) for each provider.
PROVIDER_NAMES = {
    "ollama": "ollama",
    "openai_compatible": "openai",
    "litellm": "litellm",
    "jev": "jev",
    "kokoro": "kokoro",
    "chatterbox": "chatterbox",
    "comfyui": "comfyui",
    "command": "command",
    "faster_whisper": "faster_whisper",
}

F = TypeVar("F")


def model_attributes(cfg: ModelConfig, operation: str) -> dict[str, Any]:
    """The span attributes naming the operation and model (design/current.md §8.3)."""
    return {
        "gen_ai.operation.name": operation,
        "gen_ai.provider.name": PROVIDER_NAMES[cfg.provider],
        "gen_ai.request.model": cfg.name,
        "hone.models.model_id": cfg.id,
    }


def lookup(table: Mapping[str, F], cfg: ModelConfig, what: str) -> F:
    """The provider function for `cfg`, or a `ConfigError` naming the providers that support `what`."""
    try:
        return table[cfg.provider]
    except KeyError:
        message = f"provider {cfg.provider!r} of model {cfg.id!r} does not support {what}"
        raise ConfigError(f"{message}; use one of {sorted(table)}") from None
