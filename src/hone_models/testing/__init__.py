"""Test helpers for users of hone-models: the `RecordSink` contract checker, a fake Ollama server and a
fake speech client."""

from .contracts import check_record_sink, example_span
from .fake_ollama import FakeOllama
from .fake_speech import FakeSpeech

__all__ = ["FakeOllama", "FakeSpeech", "check_record_sink", "example_span"]
