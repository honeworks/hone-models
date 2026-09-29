"""Small valid media files for the fakes (a blank PNG, a silent WAV, a packaged one-second MP4) and the
fields of an uploaded multipart form."""

from __future__ import annotations

import email.parser
import io
import struct
import wave
import zlib
from importlib import resources
from typing import Any

FAKE_AUDIO_RATE = 8000  # small files: 16 KB per second


def png(width: int = 64, height: int = 64) -> bytes:
    """A black 8-bit greyscale PNG of `width` x `height` pixels."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    pixels = zlib.compress(bytes(height * (width + 1)))  # a filter byte and a row of zeros per line
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")


def wav(duration_s: float = 1.0) -> bytes:
    """`duration_s` of 16-bit mono silence."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(FAKE_AUDIO_RATE)
        out.writeframes(b"\x00\x00" * round(duration_s * FAKE_AUDIO_RATE))
    return buffer.getvalue()


def mp4() -> bytes:
    """A packaged one-second, 64 x 64 black H.264 clip (1.7 KB)."""
    return resources.files("hone_models.testing").joinpath("data/tiny.mp4").read_bytes()


def for_save_node(key: str, values: dict[str, Any]) -> tuple[bytes, str]:
    """The file a save node writes (`key`: `images`, `audio` or `gifs`) and its extension, sized from the
    workflow's `width` / `height` or `seconds` / `duration` inputs (`values`)."""
    if key == "images":
        return png(int(values.get("width", 64)), int(values.get("height", 64))), "png"
    if key == "audio":
        return wav(float(values.get("seconds", values.get("duration", 1.0)))), "wav"
    return mp4(), "mp4"


def form_fields(body: bytes, content_type: str) -> dict[str, tuple[str | None, bytes]]:
    """A multipart/form-data body as {field name: (file name, bytes)}."""
    head = f"Content-Type: {content_type}\r\n\r\n".encode()
    message = email.parser.BytesParser().parsebytes(head + body)
    fields: dict[str, tuple[str | None, bytes]] = {}
    for part in message.walk():
        name = part.get_param("name", header="content-disposition")
        if name:
            payload = part.get_payload(decode=True)
            fields[str(name)] = (part.get_filename(), payload if isinstance(payload, bytes) else b"")
    return fields
