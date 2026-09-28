"""Image sizes from file headers (PNG, JPEG, GIF, WebP), for the context budget (change 0013).

No imaging library: the core depends on the standard library, pydantic and httpx only.
"""

from __future__ import annotations

import base64
import binascii
import struct
from pathlib import Path
from typing import Any

HEAD_BYTES = 256 * 1024  # a JPEG's size marker can follow large metadata blocks


def image_size(part: dict[str, Any]) -> tuple[int, int] | None:
    """(width, height) of an image part (`path` or `data_b64`), or `None` when it cannot be read."""
    try:
        if "data_b64" in part:
            data = base64.b64decode(str(part["data_b64"])[: HEAD_BYTES * 4 // 3 + 4])
        else:
            with Path(str(part["path"])).open("rb") as fh:
                data = fh.read(HEAD_BYTES)
    except (OSError, KeyError, ValueError, binascii.Error):
        return None
    return header_size(data)


def header_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from the first bytes of a PNG, JPEG, GIF or WebP file."""
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", data[16:24])
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return struct.unpack("<HH", data[6:10])
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return _webp(data)
        if data[:2] == b"\xff\xd8":
            return _jpeg(data)
    except struct.error:
        return None
    return None


def _webp(data: bytes) -> tuple[int, int] | None:
    kind = data[12:16]
    if kind == b"VP8 ":
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if kind == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if kind == b"VP8X":
        return int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1
    return None


def _jpeg(data: bytes) -> tuple[int, int] | None:
    at = 2
    while at + 9 < len(data):
        if data[at] != 0xFF:
            return None
        marker = data[at + 1]
        length = struct.unpack(">H", data[at + 2 : at + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # start of frame
            height, width = struct.unpack(">HH", data[at + 5 : at + 9])
            return width, height
        at += 2 + length
    return None
