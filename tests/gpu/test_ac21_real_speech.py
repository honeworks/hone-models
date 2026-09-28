"""AC-21 [real]: Kokoro-82M speaks one sentence into a non-silent WAV and frees the GPU afterwards.

Run: scripts/gpu-lock.sh uv run pytest -m gpu -k speech   (first run downloads about 330 MB)
"""

import array
import importlib
import importlib.util
import math
import os
import wave
from pathlib import Path
from typing import Any

import pytest

import hone_models as mk

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(importlib.util.find_spec("kokoro") is None, reason="needs the `speech` extra"),
]
# Read at import, before the autouse fixture points HOME at a temporary folder: the real download cache.
HF_HOME = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")


def rms(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        samples = array.array("h", wav.readframes(wav.getnframes()))
    return math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768


def test_ac21_kokoro_speaks_and_frees_the_gpu(
    gpu_lock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch: Any = importlib.import_module("torch")  # only with the extra
    monkeypatch.setenv("HF_HOME", HF_HOME)
    sink = mk.records.MemorySink()
    tts = mk.speech("kokoro-82m", sink=sink)
    text = "Attention lets every word look at every other word. That is the whole trick."
    r = tts.synthesize(text, voice="am_michael", out=tmp_path / "line.wav")
    assert r.sample_rate == 24000
    assert 2.0 < r.duration_s < 15.0
    assert rms(r.path) > 0.01  # speech, not silence
    assert sink.spans[0]["status"]["code"] == "ok"
    f = tts.synthesize("Hello from a British voice.", voice="bf_emma", speed=1.3, out=tmp_path / "b.wav")
    assert f.duration_s > 0.5
    assert rms(f.path) > 0.01
    if torch.cuda.is_available():
        # The model (about 330 MB of weights) was freed after each call; only torch's own small
        # per-process caches may stay allocated (cuBLAS workspaces, about 10 MB).
        assert torch.cuda.memory_allocated() < 32 * 1024 * 1024
