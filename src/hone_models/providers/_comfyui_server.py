"""The ComfyUI server a `comfyui` entry uses: is it up, starting it for a session, freeing it (0015 §2).

A plain call only uses a running server. A session (`client.session()` or `mk.session("comfyui")`) may
start one when none answers and `HONE_COMFYUI_START` holds a command line: it runs in the foreground as
a child in its own process group, with the clean environment of `mk.session("ollama")`, and is stopped
at the end of the block. A server that was already running is never stopped; a remote one is never
started.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from urllib.parse import urlparse

import httpx

from .. import _comfyui_loaded, _gpu_room  # module imports: _gpu_room imports the providers package
from ..errors import ConfigError, ProviderError
from ..records import log
from ..registry import COMFYUI_URL, LOCAL_HOSTS

UNSAFE_ENV = ("VIRTUAL_ENV", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME")  # as for `ollama serve`
START_TIMEOUT_S = 300.0
STOP_TIMEOUT_S = 10.0


def default_url() -> str:
    """`HONE_COMFYUI_URL`, else `http://127.0.0.1:8188`."""
    return (os.environ.get("HONE_COMFYUI_URL") or COMFYUI_URL).rstrip("/")


def healthy(url: str) -> bool:
    """True when a ComfyUI server answers `GET /system_stats` at `url`."""
    try:
        return httpx.get(f"{url}/system_stats", timeout=2).is_success
    except httpx.HTTPError:
        return False


def require(url: str) -> None:
    """Raise `ProviderError` when no server answers at `url` (a plain call never starts one)."""
    if not healthy(url):
        raise ProviderError(
            f"no ComfyUI server answers at {url}: start ComfyUI, or wrap the calls in a session "
            "(`with client.session():` or `with mk.session('comfyui'):`) with HONE_COMFYUI_START set to "
            "the command that starts it"
        )


def free(url: str) -> bool:
    """`POST /free` and clear the server's loaded-models list; False, logged, when the server did not
    take it."""
    try:
        release(url)()
    except httpx.HTTPError as exc:
        log.warning("hone-models: could not free ComfyUI at %s: %s", url, exc)
        return False
    return True


def release(url: str) -> Callable[[], None]:
    """`mk.gpu.comfyui_free(url)` that also clears the server's list in `comfyui-loaded.json`."""

    def free_and_forget() -> None:
        _gpu_room.comfyui_free(url)()
        _comfyui_loaded.clear(url)

    return free_and_forget


def register_release(url: str) -> None:
    """Let a later lease that is short of memory make this server let go (change 0005)."""
    _gpu_room.on_short(f"comfyui:{url}", release(url))


@contextmanager
def running(url: str) -> Generator[bool]:
    """Use the server at `url`, else start it from `HONE_COMFYUI_START`; yields whether it was started
    here, and stops it at the end only then."""
    if healthy(url):
        yield False
        return
    if urlparse(url).hostname not in LOCAL_HOSTS:
        raise ProviderError(f"the ComfyUI server at {url} is not answering (it is remote: not started here)")
    command = os.environ.get("HONE_COMFYUI_START", "").strip()
    if not command:
        raise ProviderError(
            f"no ComfyUI server answers at {url} and HONE_COMFYUI_START is not set: start ComfyUI, or set "
            "HONE_COMFYUI_START to the command line that starts it (e.g. '<ComfyUI>/start.sh --port 8188')"
        )
    argv = shlex.split(command)
    argv[0] = os.path.expanduser(argv[0])
    env = {k: v for k, v in os.environ.items() if k not in UNSAFE_ENV}
    try:
        proc = subprocess.Popen(  # noqa: S603 - the command the user configured
            argv, env=env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except OSError as exc:
        raise ConfigError(f"HONE_COMFYUI_START={command!r} cannot be run: {exc}") from exc
    try:
        _wait_healthy(url, proc)
        yield True
    finally:
        _stop(proc)


def _wait_healthy(url: str, proc: subprocess.Popen[bytes]) -> None:
    limit = float(os.environ.get("HONE_COMFYUI_START_S") or START_TIMEOUT_S)
    deadline = time.monotonic() + limit
    while not healthy(url):
        if proc.poll() is not None:
            raise ProviderError(
                f"HONE_COMFYUI_START exited with code {proc.returncode} before answering at {url}"
            )
        if time.monotonic() >= deadline:
            raise ProviderError(f"ComfyUI did not answer at {url} within {limit:.0f} s of starting")
        time.sleep(0.2)


def _stop(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM to the server's process group, SIGKILL after `STOP_TIMEOUT_S`."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=STOP_TIMEOUT_S)
            break
        except subprocess.TimeoutExpired:
            continue
    proc.poll()
