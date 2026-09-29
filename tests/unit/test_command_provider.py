"""The `command` provider's edges: variables, the project folder, adapters, the protocol's failure shapes."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

import hone_models as mk
from hone_models.errors import ConfigError, ProviderError
from hone_models.providers import command
from media_fixtures import LeaseRecorder


@pytest.fixture(autouse=True)
def lease(monkeypatch: pytest.MonkeyPatch) -> LeaseRecorder:
    recorder = LeaseRecorder()
    monkeypatch.setattr(mk.gpu, "GPU", recorder)
    return recorder


def client(tmp_path: Path, script: str | None = None, **entry: Any) -> mk.MediaClient:
    """A music client for an entry that runs `script` (Python code) with the test interpreter."""
    if script is not None:
        (tmp_path / "prog.py").write_text(script)
        entry.setdefault("command", [sys.executable, str(tmp_path / "prog.py"), "{request}", "{out_dir}"])
    fields = {"provider": "command", "kind": "music", **entry}
    lines = [f"{k} = {json.dumps(v)}" for k, v in fields.items() if not isinstance(v, dict)]
    tables = [f"[models.t.{k}]\n" + "\n".join(f"{a} = {json.dumps(b)}" for a, b in v.items())
              for k, v in fields.items() if isinstance(v, dict)]  # fmt: skip
    path = tmp_path / "models.toml"
    path.write_text("[models.t]\n" + "\n".join(lines) + "\n" + "\n".join(tables) + "\n")
    return mk.music("t", registry=mk.registry.load([path]), sink=mk.records.MemorySink())


WRITE_RESULT = """
import json, sys
from pathlib import Path
Path(sys.argv[2], "result.json").write_text(sys.argv[3] if len(sys.argv) > 3 else {result!r})
"""


def test_unset_variable_is_a_config_error_before_the_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lease: LeaseRecorder
) -> None:
    monkeypatch.delenv("HONE_NOPE", raising=False)
    song = client(tmp_path, command=["$HONE_NOPE/python", "{request}"])
    with pytest.raises(ConfigError, match=r"\$HONE_NOPE .* is not set"):
        song.generate("x", out=tmp_path / "a.wav")
    assert lease.calls == []
    assert not list(tmp_path.glob("*.job-*"))


def test_project_folder_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    song = client(tmp_path, command=["python", "{request}"], install={"dir_env": "HONE_X_DIR"})
    monkeypatch.delenv("HONE_X_DIR", raising=False)
    with pytest.raises(ConfigError, match=r"needs HONE_X_DIR.*models install t"):
        song.generate("x", out=tmp_path / "a.wav")
    monkeypatch.setenv("HONE_X_DIR", str(tmp_path / "missing"))
    with pytest.raises(ConfigError, match=r"not installed on this machine.*models install t"):
        song.generate("x", out=tmp_path / "a.wav")


def test_entry_without_a_command(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="needs `command`"):
        client(tmp_path).generate("x", out=tmp_path / "a.wav")


def test_cwd_that_is_not_a_folder(tmp_path: Path) -> None:
    song = client(tmp_path, command=["python"], cwd=str(tmp_path / "nowhere"))
    with pytest.raises(ConfigError, match="is not a folder"):
        song.generate("x", out=tmp_path / "a.wav")


def test_program_that_cannot_be_found(tmp_path: Path) -> None:
    song = client(tmp_path, command=["~/no-such-dir/bin/python", "{request}"])
    with pytest.raises(ConfigError, match=r"cannot run '.*/no-such-dir/bin/python'.*models install t"):
        song.generate("x", out=tmp_path / "a.wav")


def test_unknown_adapter_lists_the_packaged_ones(tmp_path: Path) -> None:
    song = client(tmp_path, command=[sys.executable, "{adapter:nope}", "{request}"])
    with pytest.raises(ConfigError, match=r"no adapter 'nope'.*'levo2'"):
        song.generate("x", out=tmp_path / "a.wav")


def test_placeholders_and_variables_are_filled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONE_Y", "why")
    cfg = client(tmp_path, command=["python"]).config
    places = {"request": "/j/request.json", "out_dir": "/j/out"}
    parts = ["{adapter:levo2}", "{request}", "--out={out_dir}", "${HONE_Y}-$HONE_Y", "{other}", "~/x"]
    filled = [command._filled(p, cfg, places) for p in parts]
    assert filled[0] == str(command.adapters_folder() / "levo2.py")
    assert filled[1:5] == ["/j/request.json", "--out=/j/out", "why-why", "{other}"]
    assert filled[5] == str(Path.home() / "x")


def test_relative_program_runs_from_cwd(tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "py").symlink_to(sys.executable)
    (tmp_path / "prog.py").write_text(WRITE_RESULT.format(result='{"error": "bad value: must be > 0"}'))
    song = client(tmp_path, command=["bin/py", "prog.py", "{request}", "{out_dir}"], cwd=str(tmp_path))
    r = song.generate("x", out=tmp_path / "a.wav")
    assert (r.error, r.error_kind) == ("bad value: must be > 0", "invalid_input")


def test_error_in_result_json_wins_over_the_exit_code(tmp_path: Path) -> None:
    script = WRITE_RESULT.format(result='{"error": "no GPU"}') + "sys.exit(2)\n"
    r = client(tmp_path, script).generate("x", out=tmp_path / "a.wav")
    assert (r.error, r.error_kind) == ("no GPU", "failed")


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ("not json", "unreadable"),
        ("[1, 2]", "must hold an object"),
        ('{"files": ["gone.flac"]}', "do not exist"),
    ],
)
def test_broken_result_json(tmp_path: Path, result: str, message: str) -> None:
    song = client(tmp_path, WRITE_RESULT.format(result=result))
    with pytest.raises(ProviderError, match=message):
        song.generate("x", out=tmp_path / "a.wav")


def test_listed_files_keep_their_order(tmp_path: Path) -> None:
    script = """
import json, sys
from pathlib import Path
out = Path(sys.argv[2])
(out / "sub").mkdir()
for name in ("z.bin", "sub/a.bin"):
    (out / name).write_bytes(b"x")
(out / "result.json").write_text(json.dumps({"files": ["z.bin", "sub/a.bin"], "error": None, "meta": {}}))
"""
    r = client(tmp_path, script).generate("x", out=tmp_path / "take")
    assert [f.path.name for f in r.files] == ["take_1.bin", "take_2.bin"]


def test_killed_by_a_signal(tmp_path: Path) -> None:
    song = client(tmp_path, "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n")
    with pytest.raises(ProviderError, match="killed by signal 9"):
        song.generate("x", out=tmp_path / "a.wav")


def test_program_that_cannot_start(tmp_path: Path) -> None:
    (tmp_path / "prog").write_text("not a program")
    (tmp_path / "prog").chmod(0o755)
    song = client(tmp_path, command=[str(tmp_path / "prog")])
    with pytest.raises(ProviderError, match="cannot start"):
        song.generate("x", out=tmp_path / "a.wav")


def test_log_tail_shows_what_a_terminal_shows(tmp_path: Path) -> None:
    log = tmp_path / "stderr.log"
    assert command.log_tail(log) is None
    log.write_bytes(b"start\nprogress 10%\rprogress 50%\rprogress 100%\n" + b"line\n" * 50)
    tail = command.log_tail(log) or ""
    assert tail.count("\n") == 39
    log.write_bytes(b"start\nprogress 10%\rprogress 100%\n\n")
    assert command.log_tail(log) == "start\nprogress 100%"
    log.write_bytes(b"\n")
    assert command.log_tail(log) is None


def test_log_tail_on_the_span_is_capped(tmp_path: Path) -> None:
    song = client(tmp_path, "import sys\nfor i in range(40):\n    print('é' * 200, file=sys.stderr)\n")
    song.generate("x", out=tmp_path / "a.wav")
    tail = song.sink.spans[0]["attributes"]["hone.models.media.log_tail"]  # type: ignore[attr-defined]
    assert len(tail.encode()) <= command.TAIL_BYTES
