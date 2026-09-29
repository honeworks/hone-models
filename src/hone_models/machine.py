"""What the machine holds, and making it hold only what is needed (design/current.md §7, change 0016).

    mk.machine.snapshot()                 # GPUs, model servers, loaded models, the GPU lock, the leases
    mk.machine.prepare(["gemma4-12b"])    # unload every other model, unless another run is using the GPU
    mk.machine.load("gemma4-12b")         # warm it up, so the first real call does not pay for loading

`Machine` has the shape of hone-select's `MachineProbe` port; `MACHINE` is the default instance and the
module-level functions are its methods. `None` always means unknown, never 0; sizes are GB with two
decimals. Nothing here waits, retries or decides whether a run should go on: that is the caller's policy.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from . import _comfyui_loaded
from ._gpu_locks import machine_lock_path, probe_lock
from ._gpu_memory import GpusReader, ProcessReader, read_gpus, read_processes
from ._gpu_room import IfBusy, check_if_busy, comfyui_free
from ._http import bearer_headers, request_json
from ._machine_read import (
    Models,
    Server,
    gb,
    gb_of_bytes,
    get_json,
    gpu_section,
    ollama_rows,
    provider,
    servers,
    tagged,
)
from ._tracing import start_span
from .errors import ProviderError
from .gpu import GPU, GpuScheduler
from .ports import RecordSink
from .providers import ollama
from .records import default_sink, log, now_iso
from .registry import ModelConfig, Registry
from .registry import load as load_registry

__all__ = ["MACHINE", "Machine", "load", "prepare", "snapshot"]

KEEP_ALIVE = "5m"  # Ollama's own default, when the registry entry sets no `defaults.keep_alive`
IN_PROCESS = ("kokoro", "chatterbox", "faster_whisper")
ON_GPU = ("comfyui", "command")  # local GPU users besides the `local` entries (change 0015)


class Machine:
    """Reads and prepares this machine's model state; `snapshot()` and `prepare()` are hone-select's
    `MachineProbe` port. The readers (`gpus`, `processes`, `scheduler`) can be replaced for tests."""

    def __init__(
        self,
        *,
        lock_path: str | Path | None = None,
        registry: Registry | None = None,
        sink: RecordSink | None = None,
        gpus: GpusReader = read_gpus,
        processes: ProcessReader = read_processes,
        scheduler: GpuScheduler | None = None,
    ) -> None:
        self._lock_path = Path(lock_path) if lock_path else None
        self._registry = registry
        self._sink = sink
        self.gpus, self.processes = gpus, processes
        self._scheduler = scheduler

    @property
    def lock_path(self) -> Path:
        """The machine-wide lock file (default `HONE_GPU_LOCK`, else `/tmp/honeworks-gpu.lock`)."""
        return self._lock_path or machine_lock_path()  # resolved on every use, like the ledger

    def snapshot(self) -> dict[str, Any]:
        """What the machine holds now, as JSON-able data (design/current.md §10). No side effects."""
        found = servers(self._registry or load_registry())
        return {
            "time": now_iso(),
            "gpus": gpu_section(self.gpus, self.processes),
            "servers": [server for server, _ in found],
            "loaded_models": [m for _, models in found for m in models],
            "gpu_lock": probe_lock(self.lock_path),
            "leases": self._leases(),
        }

    def prepare(self, needed: Sequence[str], *, if_busy: IfBusy = "block") -> dict[str, Any]:
        """Make the model servers hold only the `needed` registry ids; report what was done and the state
        after. With another process using the GPU (the lock, or a lease), `if_busy="block"` unloads
        nothing and `"unload"` unloads anyway; both report `blocked_by`. Never loads, never waits."""
        check_if_busy(if_busy)
        registry = self._registry or load_registry()
        configs = [registry.get(model_id) for model_id in needed]
        attrs = {"hone.models.machine.needed": list(needed), "hone.models.machine.if_busy": if_busy}
        with start_span(
            "hone.models.machine.prepare", self._sink or default_sink(), attrs, kind="internal"
        ) as span:
            result: dict[str, Any] = {"needed": list(needed), **self._prepare(registry, configs, if_busy)}
            for key in ("unloaded", "released", "blocked_by", "errors", "missing"):
                span["attributes"][f"hone.models.machine.{key}"] = result[key]
            if result["errors"]:
                span["status"] = {"code": "error", "message": "; ".join(e["error"] for e in result["errors"])}
        return result

    def _prepare(self, registry: Registry, configs: list[ModelConfig], if_busy: IfBusy) -> dict[str, Any]:
        before = servers(registry)
        errors = _unanswered(before)
        blocked_by = self._blocked_by()
        unloaded: list[dict[str, Any]] = []
        released: list[str] = []
        if blocked_by:
            log.warning(
                "hone-models: machine.prepare found the GPU in use: %s (if_busy=%r)", blocked_by, if_busy
            )
        if not blocked_by or if_busy == "unload":
            unloaded, released = _clean(before, configs, errors)
        after = servers(registry)
        errors += [e for e in _unanswered(after) if e not in errors]
        loaded = [m for _, models in after for m in models]
        return {
            "blocked_by": blocked_by,
            "unloaded": unloaded,
            "released": released,
            "if_busy": if_busy,
            "errors": errors,
            "missing": [c.id for c in configs if _is_missing(c, loaded)],
            "loaded_models": loaded,
            "need_gb": _need_gb(configs),
        }

    def load(self, model_id: str) -> dict[str, Any]:
        """Warm an Ollama model up (an empty `/api/generate` with `keep_alive`) and report where it landed:
        `vram_gb < size_gb` means partly on the CPU. Other providers: `loaded: None` with the reason."""
        cfg = (self._registry or load_registry()).get(model_id)
        attrs = {"hone.models.machine.model_id": cfg.id}
        with start_span(
            "hone.models.machine.load", self._sink or default_sink(), attrs, kind="internal"
        ) as span:
            result = _warm_up(cfg) if cfg.provider == "ollama" else _not_supported(cfg)
            for key in ("loaded", "seconds", "vram_gb"):
                span["attributes"][f"hone.models.machine.{key}"] = result[key]
            if result["loaded"] is False:
                span["status"] = {"code": "error", "message": result["error"]}
        return result

    def _leases(self) -> list[dict[str, Any]]:
        entries = (self._scheduler or GPU).status().leases
        me = os.getpid()
        return [
            {"name": e.get("name"), "pid": e["pid"], "vram_gb": gb(e["vram_mb"]), "mine": e["pid"] == me}
            for e in entries
        ]

    def _blocked_by(self) -> list[str] | None:
        """Who else is using the GPU: the machine-wide lock held by another process, other processes'
        leases. Our own leases and a lock held by our own gpu-lock.sh do not count."""
        lock = probe_lock(self.lock_path)
        found: list[str] = []
        if lock["held"] and not lock["mine"]:
            found.append(f"gpu lock {lock['path']} held by {lock['holder'] or 'an unknown holder'}")
        found += [
            f"lease {e['name']!r} (pid {e['pid']}, {e['vram_gb']} GB)"
            for e in self._leases()
            if not e["mine"]
        ]
        return found or None


def _unanswered(found: list[tuple[Server, Models]]) -> list[dict[str, Any]]:
    return [
        {"server": s["server"], "name": None, "error": s["error"]} for s, _ in found if s["running"] is None
    ]


def _clean(
    found: list[tuple[Server, Models]], configs: list[ModelConfig], errors: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Unload every Ollama model not needed; free a ComfyUI server holding anything not needed."""
    keep_names = {tagged(c.name) for c in configs if c.provider == "ollama"}
    keep_ids = {c.id for c in configs}
    unloaded: list[dict[str, Any]] = []
    released: list[str] = []
    for server, models in found:
        url, kind = server["url"], server["server"]
        if kind == "ollama":
            for m in (m for m in models if m["name"] not in keep_names):
                try:
                    ollama.unload(url, m["name"])
                    unloaded.append({"server": kind, "name": m["name"]})
                except ProviderError as exc:
                    log.warning("hone-models: machine.prepare could not unload %s: %s", m["name"], exc)
                    errors.append({"server": kind, "name": m["name"], "error": str(exc)})
        elif any(m["model_id"] not in keep_ids for m in models):
            try:
                comfyui_free(url)()
                _comfyui_loaded.clear(url)
                released.append(kind)
            except httpx.HTTPError as exc:
                log.warning("hone-models: machine.prepare could not free ComfyUI at %s: %s", url, exc)
                errors.append({"server": kind, "name": None, "error": f"POST {url}/free failed: {exc}"})
    return unloaded, released


def _is_missing(cfg: ModelConfig, loaded: list[dict[str, Any]]) -> bool:
    """Needed, loadable by a server and not loaded (hosted and in-process models are never missing)."""
    if cfg.provider == "ollama":
        return tagged(cfg.name) not in {m["name"] for m in loaded if m["server"] == "ollama"}
    if provider(cfg) == "comfyui":
        return cfg.id not in {m["model_id"] for m in loaded if m["server"] == "comfyui"}
    return False


def _need_gb(configs: list[ModelConfig]) -> float | None:
    """The registry `vram_gb` of the needed models that use this machine's GPU; `None` if one is unknown."""
    sizes = [c.capabilities.vram_gb for c in configs if c.local or provider(c) in ON_GPU]
    return None if None in sizes else round(sum(s for s in sizes if s is not None), 2)


def _warm_up(cfg: ModelConfig) -> dict[str, Any]:
    url = ollama.base_url(cfg)
    keep_alive = cfg.defaults.get("keep_alive", KEEP_ALIVE)
    payload = {"model": cfg.name, "prompt": "", "keep_alive": keep_alive, "stream": False}
    result: dict[str, Any] = {"model_id": cfg.id, "loaded": False, "seconds": None, "size_gb": None}
    result |= {"vram_gb": None, "error": None}
    start = time.monotonic()
    try:
        request_json(
            "POST",
            f"{url}/api/generate",
            payload=payload,
            headers=bearer_headers(cfg),
            timeout=cfg.max_timeout_s,
        )
    except ProviderError as exc:
        return result | {"seconds": round(time.monotonic() - start, 2), "error": str(exc)}
    ollama.loaded.add((url, cfg.name))  # this process loaded it: a short lease may unload it
    result |= {"loaded": True, "seconds": round(time.monotonic() - start, 2)}
    _, data, _ = get_json(f"{url}/api/ps")
    row = next((m for m in ollama_rows(data) or [] if m.get("name") == tagged(cfg.name)), None)
    if row is not None:
        result |= {"size_gb": gb_of_bytes(row.get("size")), "vram_gb": gb_of_bytes(row.get("size_vram"))}
    return result


def _not_supported(cfg: ModelConfig) -> dict[str, Any]:
    in_process = cfg.provider in IN_PROCESS or cfg.kind in ("speech", "transcription")
    how = (
        "it loads in-process and stays loaded only inside a session: use the client's session()"
        if in_process
        else "run one small job in a session instead"
    )
    error = f"not supported for {cfg.provider}: {how}"
    return {
        "model_id": cfg.id,
        "loaded": None,
        "seconds": None,
        "size_gb": None,
        "vram_gb": None,
        "error": error,
    }


MACHINE = Machine()
snapshot = MACHINE.snapshot
prepare = MACHINE.prepare
load = MACHINE.load
