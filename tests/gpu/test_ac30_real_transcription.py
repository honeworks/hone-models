"""AC-30 [real], transcription part: a Kokoro sentence transcribed by faster-whisper large-v3-turbo;
the transcript holds the sentence's words in order, and the model is freed afterwards.

Run: scripts/gpu-lock.sh uv run pytest -m gpu -k transcription
Model: `HONE_TEST_WHISPER_MODEL` (a Hugging Face repo id or a local folder; default
`deepdml/faster-whisper-large-v3-turbo-ct2`, about 1.6 GB). The test skips when it is not in the Hugging
Face cache, so it never starts a download.
"""

import importlib
import importlib.util
import os
import re
from pathlib import Path
from typing import Any

import pytest

import hone_models as mk
from whisper_fakes import registry

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(importlib.util.find_spec("kokoro") is None, reason="needs the `speech` extra"),
    pytest.mark.skipif(
        importlib.util.find_spec("faster_whisper") is None, reason="needs the `transcribe` extra"
    ),
]
# Read at import, before the autouse fixture points HOME at a temporary folder: the real download cache.
HF_HOME = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
MODEL = os.environ.get("HONE_TEST_WHISPER_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")
SENTENCE = "The lighthouse keeper counted seven ships before the morning fog rolled in."


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def in_order(wanted: list[str], heard: list[str]) -> bool:
    """Whether `wanted` is a subsequence of `heard`."""
    rest = iter(heard)
    return all(word in rest for word in wanted)


def cached(model: str) -> bool:
    if Path(model).is_dir():
        return True
    return (Path(HF_HOME) / "hub" / f"models--{model.replace('/', '--')}").is_dir()


def gpu_used_bytes() -> int | None:
    """Memory in use on GPU 0 (NVML, extra `gpu`), or `None` without NVML."""
    if importlib.util.find_spec("pynvml") is None:
        return None
    nvml: Any = importlib.import_module("pynvml")
    nvml.nvmlInit()
    try:
        return int(nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(0)).used)
    finally:
        nvml.nvmlShutdown()


def test_ac30_kokoro_sentence_transcribed_in_order(
    gpu_lock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not cached(MODEL):
        pytest.skip(f"{MODEL} is not in the Hugging Face cache ({HF_HOME}); download it first")
    monkeypatch.setenv("HF_HOME", HF_HOME)
    spoken = mk.speech("kokoro-82m", sink=mk.records.NullSink()).synthesize(
        SENTENCE, voice="am_michael", out=tmp_path / "sentence.wav"
    )
    reg = registry()
    cfg = reg.get("test-whisper").model_copy(update={"model": MODEL})
    sink = mk.records.MemorySink()
    stt = mk.Transcriber(cfg, sink)
    before = gpu_used_bytes()

    t = stt.transcribe(spoken.path, language="en", timeout_s=300)

    heard = [w for word in t.words for w in words_of(word.text)]
    assert in_order(words_of(SENTENCE), heard), (SENTENCE, t.text)
    assert all(w.end_s >= w.start_s for w in t.words)
    assert [w.start_s for w in t.words] == sorted(w.start_s for w in t.words)
    assert t.duration_s is not None
    assert abs(t.duration_s - spoken.duration_s) < 0.5
    assert (t.language, sink.spans[0]["status"]["code"]) == ("en", "ok")
    after = gpu_used_bytes()
    if before is not None and after is not None:
        # The weights (about 1.6 GB in float16) were unloaded after the call; this process keeps only its
        # CUDA context and CTranslate2's small allocator cache.
        assert after - before < 1024**3, (before, after)
