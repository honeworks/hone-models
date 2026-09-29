"""Standalone projects: one subprocess per job, in the project's own environment (change 0015 §2).

The entry's `command` is a list: `{request}`, `{out_dir}` and `{adapter:<name>}` (a script shipped in
`hone_models/data/adapters/`) are filled in, and `~` and `$VAR` / `${VAR}` are expanded, as they are in
`cwd` and the `env` values. When the entry's `install.dir_env` names the project folder variable (e.g.
`HONE_LEVO2_DIR`), it must be set to a folder, and `cwd` defaults to it. The program runs in its own
process group with the clean environment of `mk.session("ollama")` plus the entry's `env`.

The protocol: hone-models writes `request.json` (`model`, `prompt`, `seed`, `out_dir`, `inputs` with file
inputs as absolute paths, the entry's `defaults`) into a fresh job folder next to `out`; the program
writes its files into `out_dir` and may write `out_dir/result.json`, `{"files": [...], "error": ...,
"meta": {...}}`. An `error` there is `result.error`; exit code 0 gives the files (those listed, else every
file in `out_dir`); any other exit code is a `ProviderError` with the last lines of stderr. On timeout or
any exception while waiting the group gets SIGTERM, then SIGKILL after `KILL_GRACE_S`. The job folder is
removed after a success and kept after a failure, with the program's `stdout.log` and `stderr.log`.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
from importlib import resources
from pathlib import Path
from typing import Any

from .._media_files import output_paths
from .._tracing import add_event
from ..errors import ConfigError, ModelTimeout, ProviderError
from ..registry import ModelConfig
from ._comfyui_server import UNSAFE_ENV
from .common import MediaJob, MediaOutcome, MediaProvider, error_kind, listed_inputs

KILL_GRACE_S = 10.0  # between SIGTERM and SIGKILL
TAIL_LINES = 40
TAIL_BYTES = 4096  # `hone.models.media.log_tail` cap
_TOKEN = re.compile(r"\{(request|out_dir|adapter:[^{}]+)\}|\$\{(\w+)\}|\$(\w+)")


def adapters_folder() -> Path:
    """Where the packaged adapter scripts are (`{adapter:<name>}` is `<name>.py` there)."""
    return Path(str(resources.files("hone_models"))) / "data" / "adapters"


def run(job: MediaJob) -> MediaOutcome:
    """Run the entry's command for one job and collect the files it wrote."""
    cfg, attrs = job.config, job.attrs
    job_id = secrets.token_hex(6)
    job_dir = _job_folder(job.out, job_id)
    out_dir = job_dir / "out"
    argv, cwd, env = invocation(cfg, {"request": str(job_dir / "request.json"), "out_dir": str(out_dir)})
    out_dir.mkdir(parents=True)
    (job_dir / "request.json").write_text(json.dumps(request(job, out_dir), indent=2), encoding="utf-8")
    attrs["hone.models.media.job_id"] = job_id
    try:
        with job.lease():
            code = _wait(_start(argv, cwd, env, job_dir), job.timeout_s, job_id)
    finally:
        tail = log_tail(job_dir / "stderr.log")
        if tail:
            attrs["hone.models.media.log_tail"] = tail.encode()[-TAIL_BYTES:].decode(errors="ignore")
    outcome = _outcome(job, code, job_id, tail)
    if outcome.files and outcome.error is None:
        shutil.rmtree(job_dir)
    return outcome


def _job_folder(out: Path, job_id: str) -> Path:
    return out.resolve().parent / f"{out.name}.job-{job_id}"


def invocation(cfg: ModelConfig, places: dict[str, str]) -> tuple[list[str], Path | None, dict[str, str]]:
    """The entry's argv, working folder and environment, filled in and expanded; `ConfigError` for an
    unset variable, a missing project folder, an unknown adapter or a program that cannot be found."""
    if not cfg.command:
        raise ConfigError(f"model {cfg.id!r}: a command entry needs `command` (a list)")
    folder = _project_folder(cfg)
    argv = [_filled(part, cfg, places) for part in cfg.command]
    cwd = Path(_filled(cfg.cwd, cfg, places)) if cfg.cwd else folder
    if cwd is not None and not cwd.is_dir():
        raise ConfigError(f"model {cfg.id!r}: its cwd {cwd} is not a folder")
    env = {k: v for k, v in os.environ.items() if k not in UNSAFE_ENV}
    env |= {k: _filled(v, cfg, places) for k, v in (cfg.env or {}).items()}
    program = argv[0] if os.sep not in argv[0] else str((cwd or Path.cwd()) / argv[0])
    if shutil.which(program, path=env.get("PATH")) is None:
        raise ConfigError(
            f"model {cfg.id!r}: cannot run {argv[0]!r}; is the project set up? "
            f"(`hone-models models install {cfg.id}` prints the steps)"
        )
    return argv, cwd, env


def _project_folder(cfg: ModelConfig) -> Path | None:
    """The folder `install.dir_env` names, which must exist; `None` when the entry names no variable."""
    name = cfg.install.dir_env if cfg.install else None
    if not name:
        return None
    value = os.environ.get(name)
    if not value:
        raise ConfigError(
            f"model {cfg.id!r} needs {name}: set it to the project's folder "
            f"(`hone-models models install {cfg.id}` prints how to set the project up)"
        )
    folder = Path(value).expanduser()
    if not folder.is_dir():
        raise ConfigError(f"model {cfg.id!r}: {name}={value} is not a folder")
    return folder


def _filled(text: str, cfg: ModelConfig, places: dict[str, str]) -> str:
    """`text` with its placeholders filled in, `$VAR` / `${VAR}` expanded and a leading `~` expanded."""

    def one(match: re.Match[str]) -> str:
        token, name = match.group(1), match.group(2) or match.group(3)
        if token and token.startswith("adapter:"):
            return str(_adapter(token.removeprefix("adapter:"), cfg))
        if token:
            return places[token]
        value = os.environ.get(name)
        if value is None:
            raise ConfigError(f"model {cfg.id!r}: the variable ${name} in its command, cwd or env is not set")
        return value

    return os.path.expanduser(_TOKEN.sub(one, text))


def _adapter(name: str, cfg: ModelConfig) -> Path:
    script = adapters_folder() / f"{name}.py"
    if not script.is_file():
        known = sorted(p.stem for p in adapters_folder().glob("*.py"))
        raise ConfigError(f"model {cfg.id!r}: no adapter {name!r}; the packaged adapters are {known}")
    return script


def request(job: MediaJob, out_dir: Path) -> dict[str, Any]:
    """The content of `request.json`."""

    def plain(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, list):
            return [plain(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
        return value

    return {
        "model": job.config.id,
        "prompt": job.prompt,
        "seed": job.seed,
        "out_dir": str(out_dir),
        "inputs": {k: plain(v) for k, v in job.inputs.items()},
        "defaults": job.config.defaults,
    }


def _start(argv: list[str], cwd: Path | None, env: dict[str, str], job_dir: Path) -> subprocess.Popen[bytes]:
    with (job_dir / "stdout.log").open("wb") as out, (job_dir / "stderr.log").open("wb") as err:
        try:
            return subprocess.Popen(  # noqa: S603 - the program a trusted registry file names
                argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True,
            )  # fmt: skip
        except OSError as exc:
            raise ProviderError(f"cannot start {argv[0]!r}: {exc}") from exc


def _wait(proc: subprocess.Popen[bytes], timeout_s: float, job_id: str) -> int:
    """The exit code; on timeout or any exception while waiting the process group is stopped."""
    try:
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _stop(proc, job_id)
        raise ModelTimeout(
            f"command job {job_id} did not finish in {timeout_s:g} s; its processes were killed"
        ) from None
    except BaseException:
        _stop(proc, job_id)
        raise


def _stop(proc: subprocess.Popen[bytes], job_id: str) -> None:
    """SIGTERM to the process group, SIGKILL after `KILL_GRACE_S` to whatever is left of it."""
    add_event("cancelled", {"job_id": job_id})
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=KILL_GRACE_S)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)  # children that outlived the program too
    proc.wait()


def log_tail(path: Path) -> str | None:
    """The last `TAIL_LINES` lines of a log as a terminal shows them (a `\\r` rewrites its line)."""
    if not path.is_file():
        return None
    with path.open("rb") as fh:
        fh.seek(max(0, path.stat().st_size - 64 * 1024))
        text = fh.read().decode("utf-8", errors="replace")
    lines = [line.rsplit("\r", 1)[-1] for line in text.rstrip().split("\n")]
    return "\n".join(lines[-TAIL_LINES:]) or None


def _outcome(job: MediaJob, code: int, job_id: str, tail: str | None) -> MediaOutcome:
    """The files, or the error the program reported; `ProviderError` for any other non-zero exit."""
    out_dir = _job_folder(job.out, job_id) / "out"
    result = _result(out_dir)
    if result.get("error"):
        message = str(result["error"])
        return MediaOutcome([], job_id, message, error_kind(message))
    if code != 0:
        how = f"was killed by signal {-code}" if code < 0 else f"exited with code {code}"
        raise ProviderError(f"model {job.config.id!r}: job {job_id} {how}; stderr ends with:\n{tail or ''}")
    found = _files(result, out_dir)
    paths = output_paths(job.out, [p.suffix for p in found])
    for source, target in zip(found, paths, strict=True):
        shutil.move(source, target)
    return MediaOutcome(paths, job_id)


def _result(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "result.json"
    if not path.is_file():
        return {}
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ProviderError(f"the program wrote an unreadable {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ProviderError(f"{path} must hold an object with 'files', 'error' and 'meta'")
    return data  # pyright: ignore[reportUnknownVariableType]


def _files(result: dict[str, Any], out_dir: Path) -> list[Path]:
    """The files `result.json` lists (relative to `out_dir`), else every file in `out_dir` by name."""
    if "files" not in result or result["files"] is None:
        return sorted(p for p in out_dir.iterdir() if p.is_file() and p.name != "result.json")
    found = [out_dir / str(name) for name in result["files"]]
    missing = [str(p) for p in found if not p.is_file()]
    if missing:
        raise ProviderError(f"result.json lists files that do not exist: {missing}")
    return found


PROVIDER = MediaProvider(accepted=listed_inputs, run=run)
