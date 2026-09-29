"""The LeVo adapter run by the `command` provider as a subprocess, against a fake project folder
(tests/fixtures/levo2_project): request.json in, LeVo's JSONL and arguments to the project, the song out."""

import json
import sys
from pathlib import Path

import pytest

import hone_models as mk
from media_fixtures import LeaseRecorder

FIXTURES = Path(__file__).parents[1] / "fixtures"
LYRICS = "[intro-short] ; [verse] Trails wind through the forest. Trees stand tall. ; [outro-short]"


def test_levo_adapter_through_the_command_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mk.gpu, "GPU", LeaseRecorder())
    monkeypatch.setenv("HONE_TEST_PYTHON", sys.executable)
    monkeypatch.setenv("HONE_FAKE_LEVO_DIR", str(FIXTURES / "levo2_project"))
    registry = mk.registry.load([FIXTURES / "command" / "models.toml"])
    song = mk.music("test-levo", registry=registry, sink=mk.records.MemorySink())
    r = song.generate("female, pop, piano", lyrics=LYRICS, seed=42, out=tmp_path / "song.flac")
    assert r.error is None, r.error
    assert r.path == tmp_path / "song.flac"
    seen = json.loads((tmp_path / "song.flac").read_text())
    assert seen["item"] == {"idx": "take", "gt_lyric": LYRICS, "descriptions": "female, pop, piano"}
    assert seen["seeds"] == {"numpy": 42, "torch": 42, "cuda": 42}
    project = str(FIXTURES / "levo2_project")
    assert seen["args"]["ckpt_path"] == f"{project}/songgeneration_v2_medium"
    assert (seen["args"]["generate_type"], seen["args"]["low_mem"]) == ("mixed", True)
    assert seen["args"]["use_flash_attn"] is False
    assert (seen["cudnn"], seen["offload"], seen["omp"]) == (False, True, "1")
    assert seen["resolvers"] == ["concat", "eval", "get_fname", "load_yaml"]
    assert seen["cwd"] == project
    assert seen["path"] == [f"{project}/codeclm/tokenizer", project, f"{project}/codeclm/tokenizer/Flow1dVAE"]
    assert not list(tmp_path.glob("*.job-*"))
