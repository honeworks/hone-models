"""Make the reference clips of the Chatterbox voices (change 0011) from Kokoro-82M's synthetic voices.

Chatterbox takes a voice from a short reference recording. The packaged voices are rendered by Kokoro
(Apache-2.0, synthetic voices, no real person's recording), so no one's voice is cloned. Run once, under
the GPU lock, with both speech extras installed:

    scripts/gpu-lock.sh uv run python scripts/make-expressive-voices.py

It writes `src/hone_models/data/voices/<name>.flac` (24 kHz mono, about 10 s each).
"""

from pathlib import Path

import numpy as np
import soundfile as sf

import hone_models as mk

OUT = Path(__file__).resolve().parents[1] / "src" / "hone_models" / "data" / "voices"
# Chatterbox voice name -> Kokoro voice rendering its reference clip
VOICES = {
    "warm_female": "af_heart",
    "warm_male": "am_michael",
    "british_female": "bf_emma",
    "british_male": "bm_george",
}
TEXT = (
    "Hello, and welcome. In this lesson we look at one simple idea, step by step, "
    "with a few small examples along the way. Take your time with it, and let's begin."
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tts = mk.speech("kokoro-82m", sink=mk.records.NullSink())
    for name, kokoro_voice in VOICES.items():
        wav = OUT / f"{name}.wav"
        r = tts.synthesize(TEXT, voice=kokoro_voice, out=wav)
        audio, rate = sf.read(wav, dtype="int16")
        sf.write(OUT / f"{name}.flac", np.asarray(audio), rate, subtype="PCM_16")
        wav.unlink()
        print(f"{name}: {kokoro_voice}, {r.duration_s:.1f} s")


if __name__ == "__main__":
    main()
