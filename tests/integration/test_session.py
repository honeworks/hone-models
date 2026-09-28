"""`mk.session` against a fake `ollama` executable (the real service is shared: never started here)."""

import json
import socket
import stat
import sys
from pathlib import Path

import pytest
import respx

import hone_models as mk
from hone_models.errors import ConfigError, ProviderError
from hone_models.runtime import session as session_module

FAKE_OLLAMA = """#!{python}
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

Path({env_file!r}).write_text(json.dumps({{"argv": sys.argv[1:], "env": dict(os.environ)}}))
if {exit_early}:
    sys.exit(3)
host, port = os.environ["OLLAMA_HOST"].split("//")[1].split(":")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{{"version": "0.0.0-fake"}}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


HTTPServer((host, int(port)), Handler).serve_forever()
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_ollama(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Install a fake `ollama` on PATH; returns (url, env_file, write(exit_early))."""
    url = f"http://127.0.0.1:{free_port()}"
    env_file = tmp_path / "serve-env.json"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def write(exit_early: bool = False) -> None:
        exe = bin_dir / "ollama"
        exe.write_text(
            FAKE_OLLAMA.format(python=sys.executable, env_file=str(env_file), exit_early=exit_early)
        )
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)

    write()
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("OLLAMA_HOST", url)
    for name in session_module.UNSAFE_ENV:
        monkeypatch.setenv(name, "/from/the/caller")
    return url, env_file, write


def test_starts_server_with_clean_env_and_stops_it(fake_ollama) -> None:
    url, env_file, _ = fake_ollama
    assert not session_module.healthy(url)
    with mk.session("ollama") as got:
        assert got == url
        assert session_module.healthy(url)
    assert not session_module.healthy(url)
    started = json.loads(env_file.read_text())
    assert started["argv"] == ["serve"]
    assert started["env"]["OLLAMA_HOST"] == url
    assert not set(session_module.UNSAFE_ENV) & set(started["env"])


def test_running_server_is_used_and_left_running(fake_ollama) -> None:
    url, env_file, _ = fake_ollama
    with mk.session():  # we start it ...
        with mk.session():  # ... so this one finds it running
            pass
        assert session_module.healthy(url)  # the inner session did not stop it
    env_file.unlink()
    with respx.mock(base_url=url) as mock:
        mock.get("/api/version").respond(json={"version": "real"})
        with mk.session():
            pass
    assert not env_file.exists()  # nothing was started


def test_server_that_exits_early_is_reported(fake_ollama) -> None:
    _, _, write = fake_ollama
    write(exit_early=True)
    with pytest.raises(ProviderError, match="exited with code 3"), mk.session():
        pass


def test_no_server_and_no_executable(fake_ollama, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(ProviderError, match="no `ollama` executable"), mk.session():
        pass


def test_only_ollama_sessions() -> None:
    with pytest.raises(ConfigError, match="'ollama' only"), mk.session("vllm"):
        pass


def test_remote_server_is_never_started(fake_ollama, monkeypatch: pytest.MonkeyPatch) -> None:
    _, env_file, _ = fake_ollama
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box.invalid:11434")
    with pytest.raises(ProviderError, match="remote"), mk.session():
        pass
    assert not env_file.exists()
