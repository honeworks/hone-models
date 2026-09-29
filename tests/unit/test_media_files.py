"""Output naming and file measurement for generated media (change 0015 §1)."""

import os
import shutil
import struct
import wave
import zlib
from pathlib import Path

import pytest

from hone_models import _comfyui_loaded
from hone_models._media_files import flac_duration, measure, output_paths

TINY_MP4 = Path(__file__).parents[2] / "src/hone_models/testing/data/tiny.mp4"


def png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    rows = zlib.compress(b"".join(b"\x00" + b"\x00" * width for _ in range(height)))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", rows) + chunk(b"IEND", b"")


def flac_head(rate: int, samples: int) -> bytes:
    bits = (rate << 44) | (1 << 41) | (15 << 36) | samples  # rate, 2 channels, 16 bits, samples
    return b"fLaC" + bytes([0x80, 0, 0, 34]) + b"\x00" * 10 + bits.to_bytes(8, "big") + b"\x00" * 16


def test_output_paths(tmp_path: Path) -> None:
    assert output_paths(tmp_path / "a" / "x.png", [".png"]) == [tmp_path / "a" / "x.png"]
    assert (tmp_path / "a").is_dir()  # parents created
    assert output_paths(tmp_path / "take", [".flac"]) == [tmp_path / "take.flac"]
    many = output_paths(tmp_path / "shot.png", [".png", ".png"])
    assert many == [tmp_path / "shot_1.png", tmp_path / "shot_2.png"]
    assert output_paths(tmp_path / "stems", [".wav", ".flac"]) == [
        tmp_path / "stems_1.wav",
        tmp_path / "stems_2.flac",
    ]


def test_output_paths_a_dot_in_the_name_is_not_a_suffix(tmp_path: Path) -> None:
    takes = tmp_path / "takes"
    assert output_paths(takes / "ace-step-1.5-turbo", [".flac"]) == [takes / "ace-step-1.5-turbo.flac"]
    assert output_paths(tmp_path / "clips" / "v1.2", [".mp4", ".mp4"]) == [
        tmp_path / "clips" / "v1.2_1.mp4",
        tmp_path / "clips" / "v1.2_2.mp4",
    ]
    assert output_paths(takes / "song.v2", [".flac", ".wav"]) == [
        takes / "song.v2_1.flac",
        takes / "song.v2_2.wav",
    ]


def test_output_paths_keep_a_media_suffix_or_the_providers(tmp_path: Path) -> None:
    assert output_paths(tmp_path / "take.1.WAV", [".flac"]) == [tmp_path / "take.1.WAV"]  # the caller's
    shots = output_paths(tmp_path / "shot.webp", [".png", ".png"])
    assert shots == [tmp_path / "shot_1.webp", tmp_path / "shot_2.webp"]
    assert output_paths(tmp_path / "scene.xyz", [".xyz"]) == [tmp_path / "scene.xyz"]  # the provider's


def test_measure_png_and_wav_by_content(tmp_path: Path) -> None:
    image = tmp_path / "shot.bin"  # recognised by its bytes, not its name
    image.write_bytes(png(40, 30))
    got = measure(image)
    assert (got.mime, got.width, got.height, got.duration_s) == ("image/png", 40, 30, None)
    assert got.bytes == image.stat().st_size
    assert len(got.sha256) == 64
    audio = tmp_path / "take.flac"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 12000)
    got = measure(audio)
    assert (got.mime, got.duration_s, got.width) == ("audio/wav", 1.5, None)
    assert got.record()["duration_s"] == 1.5


def test_flac_streaminfo(tmp_path: Path) -> None:
    assert flac_duration(flac_head(48000, 48000 * 3)) == 3.0
    assert flac_duration(flac_head(0, 10)) is None
    assert flac_duration(b"fLaC") is None
    song = tmp_path / "song.flac"
    song.write_bytes(flac_head(44100, 44100 * 2))
    assert (measure(song).mime, measure(song).duration_s) == ("audio/flac", 2.0)


def test_unknown_file_without_ffprobe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(TINY_MP4.read_bytes())
    got = measure(clip)
    assert (got.mime, got.width, got.height, got.duration_s) == ("video/mp4", None, None, None)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not on PATH")
def test_video_with_ffprobe(tmp_path: Path) -> None:
    got = measure(TINY_MP4)
    assert (got.width, got.height, got.duration_s) == (64, 64, 1.0)


def test_broken_files_are_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
    assert measure(bad).duration_s is None
    probe = tmp_path / "bin" / "ffprobe"
    probe.parent.mkdir()
    probe.write_text('#!/bin/sh\necho \'{"format": {"duration": "N/A"}}\'\n')
    probe.chmod(0o755)
    monkeypatch.setenv("PATH", str(probe.parent) + os.pathsep + os.environ["PATH"])
    other = tmp_path / "x.hone-unknown"
    other.write_bytes(b"nothing")
    assert (measure(other).mime, measure(other).duration_s) == (None, None)


def test_comfyui_loaded_file(isolated: Path) -> None:
    assert _comfyui_loaded.read() == {}
    _comfyui_loaded.add("http://127.0.0.1:8188/", "z-image-turbo", "job-1")
    _comfyui_loaded.add("http://127.0.0.1:8188", "ace-step", "job-2")
    data = _comfyui_loaded.read()
    assert [e["model_id"] for e in data["http://127.0.0.1:8188"]] == ["z-image-turbo", "ace-step"]
    assert set(data["http://127.0.0.1:8188"][0]) == {"model_id", "job_id", "pid", "time"}
    assert _comfyui_loaded._path() == isolated / ".hone" / "models" / "comfyui-loaded.json"
    _comfyui_loaded.clear("http://127.0.0.1:8188")
    assert _comfyui_loaded.read() == {}
    _comfyui_loaded._path().write_text("not json")
    assert _comfyui_loaded.read() == {}
