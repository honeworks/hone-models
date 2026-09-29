"""faster-whisper transcription (CTranslate2, MIT), extra `transcribe`. Imported lazily; the core never
needs it (change 0015 §2).

The model comes from the Hugging Face cache or Hugging Face (`model` is a repo id such as
`deepdml/faster-whisper-large-v3-turbo-ct2`, or a local folder). `defaults` may set `device` (`auto`,
`cuda` or `cpu`), `compute_type` (default `float16` on CUDA, `int8` on the CPU), `beam_size` (5) and
`vad_filter` (false). `Engine` holds the loaded model: the transcriber opens one per call (or once per
`session()`) and closes it afterwards, like Kokoro, so it never holds memory on the shared GPU between
calls.

On CUDA, CTranslate2 loads the CUDA 12 cuBLAS and cuDNN 9 libraries when the model is first used. They
are found on the system library path, or in the `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels, which
`check_ready` preloads (decision D-051); when neither has them it raises `ConfigError` before the GPU
lease, instead of a crash in the middle of the first call.
"""

from __future__ import annotations

import ctypes
import gc
import importlib
import sys
import time
from pathlib import Path
from typing import Any

from .._transcript import Heard, TranscriptSegment, Word
from ..errors import ConfigError, ModelTimeout
from ..registry import ModelConfig
from .common import library_errors

# soname -> (the wheel that carries it, its folder under site-packages/nvidia/)
CUDA_LIBRARIES = {
    "libcublas.so.12": ("nvidia-cublas-cu12", "cublas"),
    "libcudnn.so.9": ("nvidia-cudnn-cu12", "cudnn"),
}
_loaded: set[str] = set()  # libraries found once stay loaded for the process


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ConfigError(
            "faster-whisper transcription needs the `transcribe` extra: pip install 'hone-models[transcribe]'"
        ) from exc


def settings(cfg: ModelConfig) -> tuple[str, str]:
    """The `(device, compute_type)` the entry runs with; `auto` is CUDA when CTranslate2 sees a GPU."""
    device = str(cfg.defaults.get("device") or "auto")
    if device == "auto":
        device = "cuda" if _import("ctranslate2").get_cuda_device_count() > 0 else "cpu"
    compute = cfg.defaults.get("compute_type") or ("float16" if device == "cuda" else "int8")
    return device, str(compute)


def check_ready(cfg: ModelConfig) -> None:
    """Fail before any GPU work when the model cannot run here: the extra is missing, or it runs on CUDA
    and the CUDA 12 cuBLAS or cuDNN 9 library cannot be loaded."""
    device, _ = settings(cfg)
    if device != "cuda":
        return
    missing = [soname for soname in CUDA_LIBRARIES if not _load(soname)]
    if missing:
        wheels = " ".join(CUDA_LIBRARIES[soname][0] for soname in missing)
        raise ConfigError(
            f"faster-whisper on CUDA needs {', '.join(missing)} (CUDA 12 cuBLAS, cuDNN 9), which were not "
            f"found: `pip install {wheels}`, or put the libraries on LD_LIBRARY_PATH, or set "
            f"defaults.device = 'cpu' for {cfg.id!r} (see docs/transcription.md)"
        )


def _load(soname: str) -> bool:
    """Load `soname` globally, from the system library path or from its nvidia wheel. Once loaded, the
    later `dlopen(soname)` of CTranslate2 finds it by name."""
    if soname in _loaded:
        return True
    folder = CUDA_LIBRARIES[soname][1]
    in_wheels = [Path(p) / "nvidia" / folder / "lib" / soname for p in sys.path if p]
    for candidate in [soname, *(str(path) for path in in_wheels if path.is_file())]:
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        _loaded.add(soname)
        return True
    return False


class Engine:
    """A faster-whisper model loaded once; `close()` frees it (the CTranslate2 model is unloaded)."""

    def __init__(self, cfg: ModelConfig) -> None:
        whisper = _import("faster_whisper")
        device, compute = settings(cfg)
        self.options = {
            "beam_size": int(cfg.defaults.get("beam_size", 5)),
            "vad_filter": bool(cfg.defaults.get("vad_filter", False)),
        }
        with library_errors("faster-whisper model load"):
            self.model: Any = whisper.WhisperModel(cfg.name, device=device, compute_type=compute)

    def __call__(
        self, audio: Path, language: str | None, prompt: str | None, words: bool, deadline: float
    ) -> Heard:
        found: list[TranscriptSegment] = []
        with library_errors("faster-whisper transcription"):
            segments, info = self.model.transcribe(
                str(audio), language=language, initial_prompt=prompt, word_timestamps=words, **self.options
            )
            for segment in segments:  # decoding happens while the segments are read
                found.append(_segment(segment))
                if time.monotonic() > deadline:  # stop decoding: the rest of the file is never read
                    raise ModelTimeout(f"transcribing {audio} took too long; stopped at {found[-1].end_s} s")
        duration = getattr(info, "duration", None)
        return Heard(info.language, None if duration is None else round(float(duration), 3), found)

    def close(self) -> None:
        model, self.model = self.model, None
        ct2 = getattr(model, "model", None)
        if ct2 is not None and hasattr(ct2, "unload_model"):
            ct2.unload_model()  # frees the weights on the GPU now, not when the garbage collector runs
        del model, ct2
        gc.collect()


def _segment(segment: Any) -> TranscriptSegment:
    heard: list[Any] = list(segment.words or [])  # None without word timestamps
    words = [Word(w.word.strip(), round(w.start, 3), round(w.end, 3), round(w.probability, 4)) for w in heard]
    return TranscriptSegment(segment.text.strip(), round(segment.start, 3), round(segment.end, 3), words)
