"""Local server lifecycle (design/current.md §9), ported from OneShotStudio.

    with mk.session("ollama"):          # use the running server, else start `ollama serve`
        mk.text("gemma4-12b").complete(...)
    mk.unload("gemma4-12b")             # free the model's VRAM now

A server we started is stopped on exit; one that was already running is left alone. (Two processes
starting a session at the same moment may share the server the first one started and stops.)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Generator
from contextlib import contextmanager
from urllib.parse import urlparse

import httpx

from ..errors import ConfigError, ProviderError
from ..providers import ollama
from ..registry import LOCAL_HOSTS, Registry, load

# Variables of the calling Python environment that break the server's own libraries.
UNSAFE_ENV = ("VIRTUAL_ENV", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME")
START_TIMEOUT_S = 30.0
STOP_TIMEOUT_S = 10.0


def healthy(url: str) -> bool:
    """True when an Ollama server answers at `url`."""
    try:
        return httpx.get(f"{url}/api/version", timeout=2).is_success
    except httpx.HTTPError:
        return False


def clean_env(url: str) -> dict[str, str]:
    """The environment for `ollama serve`: ours without `UNSAFE_ENV`, listening on `url`."""
    env = {k: v for k, v in os.environ.items() if k not in UNSAFE_ENV}
    env["OLLAMA_HOST"] = url
    return env


@contextmanager
def session(provider: str = "ollama") -> Generator[str]:
    """Use the running server, else start it; stop it on exit only if we started it. Yields its URL."""
    if provider != "ollama":
        raise ConfigError(f"sessions are supported for provider 'ollama' only, not {provider!r}")
    url = ollama.base_url()
    if healthy(url):
        yield url
        return
    if urlparse(url).hostname not in LOCAL_HOSTS:
        raise ProviderError(f"the Ollama server at {url} is not answering (it is remote: not started here)")
    exe = shutil.which("ollama")
    if exe is None:
        raise ProviderError(f"no Ollama server at {url} and no `ollama` executable on PATH to start one")
    proc = subprocess.Popen(  # noqa: S603 - fixed executable from PATH
        [exe, "serve"], env=clean_env(url), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        _wait_healthy(url, proc)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _wait_healthy(url: str, proc: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + START_TIMEOUT_S
    while not healthy(url):
        if proc.poll() is not None:
            raise ProviderError(
                f"`ollama serve` exited with code {proc.returncode} before answering at {url}"
            )
        if time.monotonic() >= deadline:
            raise ProviderError(f"`ollama serve` did not answer at {url} within {START_TIMEOUT_S:.0f} s")
        time.sleep(0.2)


def unload(model_id: str, *, registry: Registry | None = None) -> None:
    """Free the model's memory now (Ollama `keep_alive: 0`)."""
    cfg = (registry or load()).get(model_id)
    if cfg.provider != "ollama":
        raise ConfigError(f"model {cfg.id!r} ({cfg.provider}) cannot be unloaded; only Ollama models can")
    ollama.unload(ollama.base_url(cfg), cfg.name)
