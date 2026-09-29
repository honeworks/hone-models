"""The HeartMuLa adapter run by the `command` provider as a subprocess, against a fake heartlib
(tests/fixtures/heartlib_project) and fake checkpoint folders: request.json in, heartlib's input and
settings to the pipeline, the song out."""

import json
import sys
from pathlib import Path

import pytest

import hone_models as mk
from media_fixtures import LeaseRecorder

FIXTURES = Path(__file__).parents[1] / "fixtures"
LYRICS = "[verse]\nA quiet light across the water\n[pre-chorus]\nHold on"


def test_heartmula_adapter_through_the_command_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mk.gpu, "GPU", LeaseRecorder())
    monkeypatch.setenv("HONE_TEST_PYTHON", sys.executable)
    monkeypatch.setenv("HONE_FAKE_HEARTLIB_DIR", str(FIXTURES / "heartlib_project"))
    monkeypatch.setenv("HONE_FAKE_COMFYUI_DIR", str(tmp_path / "comfy"))
    folder = tmp_path / "comfy" / "models" / "HeartMuLa"
    for name in ("HeartMuLa-RL-oss-3B-20260123", "HeartCodec-oss-20260123"):
        (folder / name).mkdir(parents=True)
    for name in ("tokenizer.json", "gen_config.json"):
        (folder / name).write_text("{}")
    registry = mk.registry.load([FIXTURES / "command" / "models.toml"])
    song = mk.music("test-heartmula", registry=registry, sink=mk.records.MemorySink())

    r = song.generate(
        "calm guitar, warm", lyrics=LYRICS, duration_s=10, seed=42, out=tmp_path / "takes" / "v1.2"
    )

    assert r.error is None, r.error
    assert r.path == tmp_path / "takes" / "v1.2.flac"
    seen = json.loads((tmp_path / "takes" / "v1.2.flac").read_text())
    assert seen["input"] == {
        "lyrics": "[Verse]\nA quiet light across the water\n\n[Prechorus]\nHold on",
        "tags": "calm guitar,warm",
        "cfg_scale": 1.5,
    }
    settings = {"max_audio_length_ms": 10_000, "temperature": 1.0, "topk": 50, "cfg_scale": 1.5}
    assert seen["forward"] == {**settings, "cache": 30 + 125 + 1}
    assert seen["pipeline"]["heartmula_path"] == str(folder / "HeartMuLa-RL-oss-3B-20260123")
    assert seen["pipeline"]["heartcodec_path"] == str(folder / "HeartCodec-oss-20260123")
    assert seen["pipeline"]["lazy_load"] == "True"
    assert seen["mula_4bit"]["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert seen["seeds"] == {"numpy": 42, "torch": 42, "cuda": 42}
    assert seen["events"] == ["flow to cpu", "empty_cache", "decode", "unload"]
    assert (seen["rate"], seen["format"]) == (48_000, "FLAC")
    assert seen["cwd"] == str(FIXTURES / "heartlib_project")
    assert not list((tmp_path / "takes").glob("*.job-*"))
