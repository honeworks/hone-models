"""AC-28: `mk.session("comfyui")` starts ComfyUI (a fake start command) once for two clients and stops it;
a running server is never stopped; a plain call without a server says what to do and starts nothing."""

import json
import socket
import sys
from pathlib import Path

import httpx
import pytest

import hone_models as mk
from hone_models.errors import ProviderError
from hone_models.testing import FakeComfyUI
from media_fixtures import LeaseRecorder, registry

pytestmark = pytest.mark.e2e

START_SCRIPT = """
import json, os, sys
from pathlib import Path

with Path({log!r}).open("a") as log:
    log.write(json.dumps(dict(os.environ)) + "\\n")
from hone_models.testing.fake_comfyui import main

main(sys.argv[1:])
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def answers(url: str) -> bool:
    try:
        return httpx.get(f"{url}/system_stats", timeout=2).is_success
    except httpx.HTTPError:
        return False


@pytest.fixture
def start_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    """HONE_COMFYUI_URL on a free port and HONE_COMFYUI_START running a fake ComfyUI there; each start
    appends its environment to the returned log file."""
    monkeypatch.setattr(mk.gpu, "GPU", LeaseRecorder())
    port, log = free_port(), tmp_path / "starts.jsonl"
    script = tmp_path / "start_comfyui.py"
    script.write_text(START_SCRIPT.format(log=str(log)))
    url = f"http://127.0.0.1:{port}"
    monkeypatch.setenv("HONE_COMFYUI_URL", url)
    monkeypatch.setenv("HONE_COMFYUI_START", f"{sys.executable} {script} --port {port}")
    monkeypatch.setenv("PYTHONPATH", "/from/the/caller")  # removed from the server's environment
    return url, log


def starts(log: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_ac28_session_starts_once_for_two_clients_and_stops(start_command, tmp_path: Path) -> None:
    url, log = start_command
    reg = registry()
    with mk.session("comfyui") as got:
        assert got == url
        assert answers(url)
        img = mk.image("test-image", registry=reg).generate("a lighthouse", out=tmp_path / "a.png")
        song = mk.music("test-music", registry=reg).generate("lo-fi", duration_s=2, out=tmp_path / "b")
    assert (img.error, song.error) == (None, None)
    assert song.files[0].duration_s == 2.0
    (env,) = starts(log)  # started once
    assert "PYTHONPATH" not in env
    assert not answers(url)  # stopped at the end


def test_ac28_client_session_starts_the_server(start_command, tmp_path: Path) -> None:
    url, log = start_command
    sink = mk.records.MemorySink()
    img = mk.image("test-image", registry=registry(), sink=sink)
    with img.session():
        img.generate("one", out=tmp_path / "1.png")
        img.generate("two", out=tmp_path / "2.png")
    started = [s["attributes"]["hone.models.media.server_started"] for s in sink.spans]
    assert started == [True, False]
    assert len(starts(log)) == 1
    assert not answers(url)


def test_ac28_running_server_is_never_stopped(start_command, tmp_path: Path) -> None:
    _, log = start_command
    with FakeComfyUI() as server:  # points HONE_COMFYUI_URL at itself
        with mk.session("comfyui") as got:
            assert got == server.url
            mk.image("test-image", registry=registry()).generate("x", out=tmp_path / "x.png")
        assert answers(server.url)
    assert starts(log) == []


def test_ac28_plain_call_without_a_server(start_command, tmp_path: Path) -> None:
    url, log = start_command
    img = mk.image("test-image", registry=registry())
    with pytest.raises(ProviderError, match=r"start ComfyUI, or wrap the calls in a session"):
        img.generate("x", out=tmp_path / "x.png")
    assert starts(log) == []
    assert not answers(url)
