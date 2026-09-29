"""Test helpers for users of hone-models: the `RecordSink` contract checker, fake Ollama and ComfyUI
servers, and fake speech, media and transcription clients."""

from .contracts import check_record_sink, example_span
from .fake_comfyui import FakeComfyUI
from .fake_media import FakeMedia
from .fake_ollama import FakeOllama
from .fake_speech import FakeSpeech
from .fake_transcriber import FakeTranscriber

__all__ = [
    "FakeComfyUI",
    "FakeMedia",
    "FakeOllama",
    "FakeSpeech",
    "FakeTranscriber",
    "check_record_sink",
    "example_span",
]
