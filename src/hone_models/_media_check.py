"""`hone-models models check <id>` for image, music and video entries (change 0015 §3b): one tiny job in a
session (which starts ComfyUI when `HONE_COMFYUI_START` is set and nothing answers), the output measured,
and the GPU memory sampled while it runs, so a new workflow is proven and its `vram_gb` measured.

The tiny job takes the smallest inputs that make every packaged model run: a 256x256 image, 10 s of
audio (HeartMuLa's shortest), a 1 s, 256x256 clip from a generated start frame; only the inputs the entry
takes are passed.
"""

from __future__ import annotations

import os
import struct
import threading
import time
import zlib
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ._gpu_memory import read_memory
from .media import MediaClient

TINY: dict[str, dict[str, Any]] = {
    "image": {"size": "256x256"},
    "music": {"duration_s": 10},
    "video": {"size": "256x256", "duration_s": 1},
}
PROMPTS = {
    "image": "a red apple on a wooden table, soft daylight",
    "music": "calm acoustic guitar, warm, slow",
    "video": "slow push-in, leaves moving in the wind",
}
LYRICS = "[verse]\nA quiet light across the water\n"
SAMPLE_S = 0.25


def default_out_dir() -> Path:
    """`${HONE_HOME:-.hone}/models/checks`: where the tiny outputs are written."""
    return Path(os.environ.get("HONE_HOME", ".hone")) / "models" / "checks"


def check(client: MediaClient, out_dir: Path, *, seed: int = 1) -> dict[str, Any]:
    """Run the tiny job for `client` and return what it made: time, file, measures, peak GPU memory."""
    inputs = {k: v for k, v in TINY[client.kind].items() if k in client.inputs}
    if "lyrics" in client.inputs:
        inputs["lyrics"] = LYRICS
    if "image" in client.inputs:
        inputs["image"] = start_frame(out_dir / "start-frame.png", 256, 256)
    started = time.monotonic()
    with peak_gpu_mb() as peak, client.session():
        r = client.generate(PROMPTS[client.kind], out=out_dir / client.model_id / "tiny", seed=seed, **inputs)
    first = r.files[0] if r.files else None
    return {
        "id": client.model_id,
        "kind": client.kind,
        "seconds": round(time.monotonic() - started, 1),
        "job_seconds": r.elapsed_s,
        "inputs": {k: str(v) for k, v in inputs.items()},
        "error": r.error,
        "error_kind": r.error_kind,
        **(first.record() if first else {}),
        "peak_vram_gb": round(peak["mb"] / 1024, 2) if peak["mb"] is not None else None,
    }


@contextmanager
def peak_gpu_mb() -> Generator[dict[str, int | None]]:
    """Sample GPU 0's used memory while the block runs; `["mb"]` is the peak above the start (`None`
    when the memory cannot be read). ComfyUI runs in another process, so the whole card is measured."""
    first = read_memory()
    found: dict[str, int | None] = {"mb": None}
    if first is None:
        yield found
        return
    base, top, stop = first[1], [first[1]], threading.Event()

    def sample() -> None:
        while not stop.wait(SAMPLE_S):
            now = read_memory()
            if now is not None:
                top.append(now[1])

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield found
    finally:
        stop.set()
        thread.join()
        found["mb"] = max(top) - base


def start_frame(path: Path, width: int, height: int) -> Path:
    """Write a plain PNG (a colour gradient) to use as the start frame of a tiny video job."""
    rows = b"".join(
        b"\0" + bytes(v for x in range(width) for v in (x * 255 // width, y * 255 // height, 128))
        for y in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + body)
    return path
