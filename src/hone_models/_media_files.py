"""Generated files: where they go (`output_paths`) and what they are (`measure`), change 0015 §1.

A file is recognised by its first bytes, not its name: images by the header reader of change 0013, WAV
with `wave`, FLAC from its STREAMINFO block, anything else with `ffprobe` when it is on `PATH`. What
cannot be read is `None`, never 0.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._images import HEAD_BYTES, header_size

FFPROBE_TIMEOUT_S = 30
# File suffixes that name a media file: an `out` ending in one of these keeps it (`output_paths`).
MEDIA_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif", ".bmp", ".tif", ".tiff",
    ".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac",
    ".mp4", ".webm", ".mov", ".mkv", ".avi",
})  # fmt: skip
_IMAGE_MIME = {
    b"\x89PNG": "image/png",
    b"\xff\xd8": "image/jpeg",
    b"GIF8": "image/gif",
    b"RIFF": "image/webp",
}


@dataclass(frozen=True, slots=True)
class MediaFile:
    """One output file: its path, SHA-256, size in bytes, MIME type and, when they apply and can be read,
    its width and height in pixels and its duration in seconds."""

    path: Path
    sha256: str
    bytes: int
    mime: str | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None

    def record(self) -> dict[str, Any]:
        """The span form: `{"path", "sha256", "bytes", "mime", "width", "height", "duration_s"}`."""
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.bytes,
            "mime": self.mime,
            "width": self.width,
            "height": self.height,
            "duration_s": self.duration_s,
        }


def file_digest(path: Path) -> tuple[str, int]:
    """(SHA-256 hex, size in bytes) of a file, read in blocks."""
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def file_record(path: Path) -> dict[str, Any]:
    """A file input as recorded on a span: `{"path", "sha256", "bytes"}`."""
    sha, size = file_digest(path)
    return {"path": str(path), "sha256": sha, "bytes": size}


def output_paths(out: Path, suffixes: list[str]) -> list[Path]:
    """Where each of several outputs goes: `out` for one file, else `<stem>_1<suffix>`, `<stem>_2<suffix>`,
    ...; when `out` has no file suffix, the provider's (`suffixes[i]`) is appended. Only a media suffix
    (`MEDIA_SUFFIXES`) or one of the provider's counts as `out`'s own, so a dot in a name
    (`takes/ace-step-1.5-turbo`, `clips/v1.2`) is kept. Parent folders are created."""
    out.parent.mkdir(parents=True, exist_ok=True)
    known = MEDIA_SUFFIXES | {s.lower() for s in suffixes}
    own = out.suffix if out.suffix.lower() in known else ""
    stem = out.name.removesuffix(own) if own else out.name
    paths: list[Path] = []
    for i, suffix in enumerate(suffixes, 1):
        name = stem if len(suffixes) == 1 else f"{stem}_{i}"
        paths.append(out.with_name(name + (own or suffix)))
    return paths


def measure(path: Path) -> MediaFile:
    """Hash, size, MIME type and (when readable) dimensions or duration of a written file."""
    sha, size = file_digest(path)
    with path.open("rb") as fh:
        head = fh.read(HEAD_BYTES)
    dims = header_size(head)
    if dims is not None:
        mime = next((m for magic, m in _IMAGE_MIME.items() if head.startswith(magic)), None)
        return MediaFile(path, sha, size, mime, dims[0], dims[1])
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return MediaFile(path, sha, size, "audio/wav", duration_s=_wav_duration(path))
    if head[:4] == b"fLaC":
        return MediaFile(path, sha, size, "audio/flac", duration_s=flac_duration(head))
    width, height, duration = _ffprobe(path)
    return MediaFile(path, sha, size, mimetypes.guess_type(path.name)[0], width, height, duration)


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            return round(wav.getnframes() / rate, 3) if rate else None
    except (wave.Error, EOFError):
        return None


def flac_duration(head: bytes) -> float | None:
    """Seconds from a FLAC file's STREAMINFO block (the first metadata block), or `None`."""
    if len(head) < 26 or head[4] & 0x7F != 0:  # block type 0 is STREAMINFO
        return None
    bits = int.from_bytes(head[18:26], "big")  # sample rate: 20 bits, ..., total samples: low 36 bits
    rate, samples = bits >> 44, bits & ((1 << 36) - 1)
    return round(samples / rate, 3) if rate and samples else None


def _ffprobe(path: Path) -> tuple[int | None, int | None, float | None]:
    """(width, height, duration) from `ffprobe`, each `None` when unknown or when ffprobe is missing."""
    exe = shutil.which("ffprobe")
    if exe is None:
        return None, None, None
    args = [exe, "-v", "error", "-show_entries", "format=duration:stream=width,height", "-of", "json"]
    try:
        done = subprocess.run(  # noqa: S603 - ffprobe from PATH on a file we wrote
            [*args, str(path)], capture_output=True, timeout=FFPROBE_TIMEOUT_S, check=False
        )
        data: dict[str, Any] = json.loads(done.stdout or b"{}")
        streams: list[dict[str, Any]] = data.get("streams", [])
        video = next((s for s in streams if s.get("width")), dict[str, Any]())
        duration = data.get("format", {}).get("duration")
        seconds = round(float(duration), 3) if duration else None
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError, AttributeError):
        return None, None, None
    return video.get("width"), video.get("height"), seconds
