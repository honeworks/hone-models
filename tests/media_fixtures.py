"""Shared helpers for the generation tests: the test registry, a reference image, a lease recorder."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hone_models as mk

COMFYUI_FIXTURES = Path(__file__).parent / "fixtures" / "comfyui"


def registry() -> mk.registry.Registry:
    """The packaged registry plus `test-image`, `test-music` and `test-video` (tests/fixtures/comfyui)."""
    return mk.registry.load([COMFYUI_FIXTURES / "models.toml"])


def png_file(path: Path, width: int = 8, height: int = 8) -> Path:
    """Write a small valid PNG and return its path."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    pixels = zlib.compress(bytes(height * (width + 1)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")
    )
    return path


class LeaseRecorder:
    """A GPU lease that records `(name, vram_gb)` and never waits."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []
        self.held = 0

    @contextmanager
    def lease(
        self, name: str, vram_gb: float, *, timeout_s: float | None = None, trace: Any = None
    ) -> Iterator[None]:
        self.calls.append((name, vram_gb))
        self.held += 1
        try:
            yield
        finally:
            self.held -= 1
