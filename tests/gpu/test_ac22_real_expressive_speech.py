"""AC-22 [real]: Chatterbox says the same line calmly and excitedly; both are speech, and the excited take
varies more in pitch, is louder and higher. The GPU is freed afterwards.

Run: scripts/gpu-lock.sh uv run pytest -m gpu -k expressive -s   (first run downloads about 3 GB)
"""

import importlib
import importlib.util
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import hone_models as mk

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(importlib.util.find_spec("chatterbox") is None, reason="needs the `expressive` extra"),
]
# Read at import, before the autouse fixture points HOME at a temporary folder: the real download cache.
HF_HOME = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
LINE = "We finished the project on time, and the results are better than anyone expected."


def prosody(path: Path) -> dict[str, float]:
    """rms: overall level; energy_db_sd: spread of 40 ms frame loudness (dB) over speech frames;
    pitch_st_sd: spread of an autocorrelation pitch estimate (semitones) over voiced frames."""
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16) / 32768.0
    frame, hop = int(0.04 * rate), int(0.01 * rate)
    frames = np.stack([audio[i : i + frame] for i in range(0, len(audio) - frame, hop)])
    level = np.sqrt((frames**2).mean(axis=1))
    speech = frames[level > level.max() * 0.05]
    db = 20 * np.log10(level[level > level.max() * 0.05])
    lo, hi = rate // 400, rate // 70  # 70 to 400 Hz
    pitches = []
    for raw in speech:
        f = (raw - raw.mean()) * np.hanning(frame)
        ac = np.fft.irfft(np.abs(np.fft.rfft(f, 2 * frame)) ** 2)[:frame]
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[0] > 0 and ac[lag] / ac[0] > 0.5:  # clearly periodic: voiced
            pitches.append(rate / lag)
    semitones = 12 * np.log2(np.array(pitches) / np.median(pitches))
    return {
        "rms": float(np.sqrt((audio**2).mean())),
        "energy_db_sd": float(db.std()),
        "pitch_st_sd": float(semitones.std()),
        "pitch_hz": float(np.median(pitches)),
    }


def test_ac22_chatterbox_calm_and_excited_differ(
    gpu_lock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch: Any = importlib.import_module("torch")  # only with the extra
    monkeypatch.setenv("HF_HOME", HF_HOME)
    sink = mk.records.MemorySink()
    tts = mk.speech("chatterbox", sink=sink)
    with tts.session():  # one model load for both takes
        calm = tts.synthesize(
            LINE, voice="warm_female", emotion="calm", intensity=0.0, out=tmp_path / "c.wav"
        )
        excited = tts.synthesize(
            LINE, voice="warm_female", emotion="excited", intensity=1.0, out=tmp_path / "x.wav"
        )
    stats = {"calm": prosody(calm.path), "excited": prosody(excited.path)}
    print("\nprosody:", {k: {n: round(v, 3) for n, v in s.items()} for k, s in stats.items()})
    for r, s in zip((calm, excited), stats.values(), strict=True):
        assert r.sample_rate == 24000
        assert 2.0 < r.duration_s < 20.0
        assert s["rms"] > 0.01  # speech, not silence
    # Measured on CPU (seed 0): pitch SD 2.9 -> 3.8 semitones, level 0.06 -> 0.11, median pitch 195 -> 279 Hz.
    # The spread of frame loudness moves both ways between voices, so it is printed, not asserted.
    calm_s, excited_s = stats["calm"], stats["excited"]
    assert excited_s["pitch_st_sd"] > calm_s["pitch_st_sd"]  # a livelier melody
    assert excited_s["rms"] > calm_s["rms"]  # more energy
    assert excited_s["pitch_hz"] > calm_s["pitch_hz"]  # a raised voice
    assert [s["attributes"]["hone.models.speech.emotion"] for s in sink.spans] == ["calm", "excited"]
    assert [s["attributes"]["hone.models.speech.loaded"] for s in sink.spans] == [True, False]
    assert all(s["status"]["code"] == "ok" for s in sink.spans)
    if torch.cuda.is_available():
        assert torch.cuda.memory_allocated() < 32 * 1024 * 1024  # the 3 GB model was freed after the session
