"""Making room on the GPU for a lease, and saying why there is none (design/current.md §7).

Besides unloading Ollama models, a lease that is short of memory calls the release hooks registered with
`on_short` (change 0005), for GPU users the lease cannot unload itself: a ComfyUI server
(`comfyui_free(url)`), torch's cache (`torch_empty_cache`), anything else with a "let go" call.
"""

from __future__ import annotations

import gc
import importlib
import os
from collections.abc import Callable

import httpx

from . import _comfyui_loaded
from .errors import CapabilityError, ProviderError
from .providers import ollama
from .records import log

_HOOKS: dict[str, Callable[[], object]] = {}


def on_short(name: str, release: Callable[[], object] | None) -> None:
    """Call `release()` when a lease in this process is short of memory: once per lease, after idle Ollama
    models are unloaded and before it waits, in registration order. Registering `name` again replaces its
    hook; `None` removes it. A hook that raises is logged and skipped."""
    _HOOKS.pop(name, None)
    if release is not None:
        _HOOKS[name] = release


def run_hooks() -> list[str]:
    """Call every release hook; the names of those that ran without raising."""
    done: list[str] = []
    for name, release in list(_HOOKS.items()):
        try:
            release()
            done.append(name)
        except Exception as exc:  # a hook is user code: log it and go on making room
            log.warning("hone-models: GPU release hook %r failed: %s", name, exc)
    return done


def comfyui_free(url: str) -> Callable[[], object]:
    """A release hook asking the ComfyUI server at `url` to unload its models (`POST /free`); the server's
    list of loaded models (`comfyui-loaded.json`, change 0015) is cleared once it has."""

    def free() -> None:  # ComfyUI answers 200 with an empty body
        payload = {"unload_models": True, "free_memory": True}
        httpx.post(f"{url.rstrip('/')}/free", json=payload, timeout=30).raise_for_status()
        _comfyui_loaded.clear(url)

    return free


def torch_empty_cache() -> None:
    """A release hook returning torch's cached, unused GPU memory in this process (no-op without torch)."""
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_idle(unload_others: bool) -> list[str]:
    """Unload the Ollama models this process loaded (every running one with `unload_others`)."""
    targets = set(ollama.loaded)
    try:
        if unload_others:
            targets |= ollama.running()
    except ProviderError as exc:
        log.warning("hone-models: could not list running Ollama models: %s", exc)
    done: list[str] = []
    for url, model in sorted(targets):
        try:
            ollama.unload(url, model)
            done.append(model)
        except ProviderError as exc:
            log.warning("hone-models: could not unload %s: %s", model, exc)
    return done


def stalled(
    name: str,
    need: int,
    free: int,
    *,
    stall_s: float,
    processes: dict[int, int] | None,
    unload_others: bool,
    nested_in: str | None = None,
) -> CapabilityError:
    """The error for a lease no one can end: who holds the memory (`processes`, {pid: MB}) and what
    to do."""
    others = {p: mb for p, mb in (processes or {}).items() if p != os.getpid()}
    holders = ", ".join(f"pid {p} ({mb} MB)" for p, mb in sorted(others.items())) or "other programs"
    message = (
        f"GPU lease {name!r} needs {need} MB but only {free} MB has been free for "
        f"{stall_s:.0f} s with no other lease held, so waiting will not help: lower vram_gb or "
        f"free the GPU memory held by {holders}"
    )
    if nested_in:
        message += (
            f"; that is on top of what this process's lease {nested_in!r} around it reserved (lower the "
            "outer vram_gb when the inner client leases by itself)"
        )
    if not unload_others:
        try:
            idle = sorted(model for _, model in ollama.running())
        except ProviderError:
            idle = []
        if idle:
            message += f"; Ollama still holds {idle} (GpuScheduler(unload_others=True) unloads them)"
    return CapabilityError(message)
