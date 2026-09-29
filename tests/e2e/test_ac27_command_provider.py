"""AC-27: the `command` provider with a fake project - success, an error in result.json, a non-zero exit
and a job that hangs."""

import _thread
import json
import sys
import threading
import time
from pathlib import Path

import pytest

import hone_models as mk
from hone_models.errors import ModelTimeout, ProviderError
from hone_models.providers import command
from media_fixtures import LeaseRecorder

pytestmark = pytest.mark.e2e

FIXTURES = Path(__file__).parents[1] / "fixtures" / "command"


@pytest.fixture(autouse=True)
def lease(monkeypatch: pytest.MonkeyPatch) -> LeaseRecorder:
    recorder = LeaseRecorder()  # never the real GPU in the default suite
    monkeypatch.setattr(mk.gpu, "GPU", recorder)
    return recorder


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    folder = tmp_path / "seen"
    folder.mkdir()
    monkeypatch.setenv("HONE_TEST_PYTHON", sys.executable)
    monkeypatch.setenv("HONE_FAKE_PROJECT_DIR", str(FIXTURES))
    monkeypatch.setenv("HONE_FAKE_SEEN", str(folder))
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")  # the clean environment drops it
    return folder


@pytest.fixture
def song(seen: Path) -> mk.MediaClient:
    registry = mk.registry.load([FIXTURES / "models.toml"])
    return mk.music("test-song", registry=registry, sink=mk.records.MemorySink())


def spans(client: mk.MediaClient) -> list[dict]:
    return client.sink.spans  # type: ignore[attr-defined]


def alive(pid: int) -> bool:
    """True while `pid` runs (a zombie counts as gone)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def test_ac27_success_collects_the_files(
    song: mk.MediaClient, seen: Path, tmp_path: Path, lease: LeaseRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ref.wav").write_bytes(b"RIFF")
    r = song.generate(
        "dark trap",
        duration_s=2,
        seed=11,
        source="ref.wav",
        references=["ref.wav"],
        out=Path("takes/take.wav"),
    )
    assert r.error is None
    assert r.path == Path("takes/take.wav")
    assert r.files[0].duration_s == pytest.approx(2.0)
    assert r.files[0].mime == "audio/wav"
    assert lease.calls == [("test-song", 5.5)]
    assert list(Path("takes").iterdir()) == [r.path]  # the job folder is gone after a success
    request = json.loads((seen / "seen.json").read_text())
    assert request["cwd"] == str(FIXTURES)  # cwd defaults to the project folder variable
    assert request["env"] == {"VIRTUAL_ENV": None, "PYTHONPATH": None, "FAKE_SEEN": str(seen)}
    sent = request["request"]
    assert (sent["model"], sent["prompt"], sent["seed"]) == ("test-song", "dark trap", 11)
    ref = str(tmp_path / "ref.wav")
    assert sent["inputs"] == {"mode": "ok", "duration_s": 2, "source": ref, "references": [ref]}
    assert Path(sent["out_dir"]).is_absolute()
    (span,) = spans(song)
    attrs = span["attributes"]
    assert span["name"] == "hone.models.music"
    assert attrs["gen_ai.provider.name"] == "command"
    assert attrs["hone.models.media.job_id"] == r.job_id
    assert attrs["hone.models.media.outputs"][0]["sha256"] == r.files[0].sha256
    assert attrs["hone.models.media.log_tail"] == "fake project: mode ok"


def test_ac27_files_without_result_json(song: mk.MediaClient, tmp_path: Path) -> None:
    r = song.generate("dark trap", mode="unlisted", out=tmp_path / "take.wav")
    assert [f.path.name for f in r.files] == ["take_1.wav", "take_2.wav"]  # a.wav, b.wav in name order


def test_ac27_error_in_result_json_is_a_result(song: mk.MediaClient, tmp_path: Path) -> None:
    r = song.generate("dark trap", mode="error", out=tmp_path / "take.wav")
    assert (r.files, r.error_kind) == ([], "out_of_memory")
    assert "CUDA out of memory" in (r.error or "")
    (span,) = spans(song)
    assert span["status"]["code"] == "error"
    job_folders = [p for p in tmp_path.iterdir() if ".job-" in p.name]
    assert len(job_folders) == 1  # kept after a failure, for a look at the request and the logs
    assert (job_folders[0] / "request.json").is_file()


def test_ac27_exit_without_files_is_no_output(song: mk.MediaClient, tmp_path: Path) -> None:
    r = song.generate("dark trap", mode="empty", out=tmp_path / "take.wav")
    assert (r.files, r.error_kind) == ([], "no_output")


def test_ac27_non_zero_exit_raises_with_the_stderr_tail(song: mk.MediaClient, tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="exited with code 3") as caught:
        song.generate("dark trap", mode="crash", out=tmp_path / "take.wav")
    message = str(caught.value)
    assert "stderr line 59" in message
    assert "stderr line 20" in message
    assert "stderr line 19" not in message  # the last 40 lines only
    (span,) = spans(song)
    assert span["status"]["code"] == "error"
    assert span["attributes"]["hone.models.media.log_tail"].splitlines()[-1] == "stderr line 59"


@pytest.mark.parametrize("mode", ["hang", "stubborn"])
def test_ac27_hang_kills_the_process_group(
    song: mk.MediaClient, seen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr(command, "KILL_GRACE_S", 0.5)
    started = time.monotonic()
    with pytest.raises(ModelTimeout, match="killed"):
        song.generate("dark trap", mode=mode, timeout_s=2, out=tmp_path / "take.wav")
    assert time.monotonic() - started < 10
    pids = json.loads((seen / "pids.json").read_text())
    deadline = time.monotonic() + 5
    while any(alive(p) for p in pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(alive(p) for p in pids)  # the program and its child
    (span,) = spans(song)
    assert [e["name"] for e in span["events"]] == ["cancelled"]
    assert span["attributes"]["hone.models.media.log_tail"].endswith("working...")


def test_ac27_interrupt_while_waiting_kills_the_process_group(
    song: mk.MediaClient, seen: Path, tmp_path: Path
) -> None:
    timer = threading.Timer(1.0, _thread.interrupt_main)
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            song.generate("dark trap", mode="hang", out=tmp_path / "take.wav")
    finally:
        timer.cancel()
    pids = json.loads((seen / "pids.json").read_text())
    deadline = time.monotonic() + 5
    while any(alive(p) for p in pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(alive(p) for p in pids)
    assert [e["name"] for e in spans(song)[0]["events"]] == ["cancelled"]
