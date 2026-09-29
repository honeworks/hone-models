"""Reading what the machine holds for `mk.machine` (change 0016): GPUs, model servers and their models.

Every reader reports "unknown" as `None` (never 0) and never raises for a missing reader or a server that
does not answer: those are facts for the caller. Sizes are GB with two decimals.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import httpx

from . import _comfyui_loaded
from ._gpu_memory import GpusReader, ProcessReader
from .providers import ollama
from .registry import ModelConfig, Registry

GIB = 1024**3
TIMEOUT_S = 2.0  # a snapshot costs at most a few seconds when a server hangs
COMFYUI_URL = "http://127.0.0.1:8188"
Server = dict[str, Any]
Models = list[dict[str, Any]]


def gb(mb: float | None) -> float | None:
    """MB as GB with two decimals; `None` stays `None`."""
    return None if mb is None else round(mb / 1024, 2)


def gb_of_bytes(size: Any) -> float | None:
    return round(size / GIB, 2) if isinstance(size, int | float) else None


def tagged(name: str) -> str:
    """The name Ollama's `/api/ps` uses: `name:latest` when `name` has no tag."""
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


def provider(cfg: ModelConfig) -> str:
    """The entry's provider as a plain string (`comfyui` and `command` arrive with change 0015)."""
    return str(cfg.provider)


def get_json(url: str) -> tuple[bool | None, dict[str, Any], str | None]:
    """(running, JSON object, error): `False` when the connection is refused, `None` with the error when
    the server timed out or answered with an error or something other than a JSON object."""
    try:
        resp = httpx.get(url, timeout=TIMEOUT_S)
    except httpx.ConnectError:
        return False, {}, None
    except httpx.HTTPError as exc:
        return None, {}, f"{type(exc).__name__}: {exc}"
    if resp.status_code >= 400:
        return None, {}, f"{url} returned HTTP {resp.status_code}: {resp.text[:200]}"
    try:
        data: Any = resp.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return None, {}, f"{url} returned no JSON object: {resp.text[:200]!r}"
    return True, cast(dict[str, Any], data), None


def gpu_section(read: GpusReader, processes: ProcessReader) -> list[dict[str, Any]] | None:
    """The `gpus` section; per-process memory is known for GPU 0 only (the lease's GPU)."""
    found = read()
    if found is None:
        return None
    procs = processes()
    out: list[dict[str, Any]] = []
    for gpu in found:
        total, used = gpu["memory_total_mb"], gpu["memory_used_mb"]
        holders = None
        if gpu["index"] == 0 and procs is not None:
            holders = [
                {"pid": p, "memory_gb": gb(mb), "mine": p == os.getpid()} for p, mb in sorted(procs.items())
            ]
        out.append(
            {
                "index": gpu["index"],
                "name": gpu["name"],
                "memory_total_gb": gb(total),
                "memory_used_gb": gb(used),
                "memory_free_gb": None if total is None or used is None else gb(total - used),
                "utilization_pct": gpu["utilization_pct"],
                "processes": holders,
            }
        )
    return out


def servers(registry: Registry) -> list[tuple[Server, Models]]:
    """Each model server with the models it holds: Ollama, then every ComfyUI server the registry names."""
    loaded = _comfyui_loaded.read() if comfyui_urls(registry) else {}
    return [read_ollama(registry)] + [read_comfyui(url, registry, loaded) for url in comfyui_urls(registry)]


def read_ollama(registry: Registry) -> tuple[Server, Models]:
    url = ollama.base_url()
    running, data, error = get_json(f"{url}/api/ps")
    rows = ollama_rows(data)
    if running and rows is None:
        running, error = None, f"{url}/api/ps returned no model list"
    server: Server = {"server": "ollama", "url": url, "running": running, "error": error}
    entries = sorted(registry.models.values(), key=lambda c: c.id, reverse=True)  # the first id wins
    ids = {tagged(c.name): c.id for c in entries if c.provider == "ollama"}
    models = [
        {
            "server": "ollama",
            "name": m.get("name"),
            "model_id": ids.get(str(m.get("name"))),
            "size_gb": gb_of_bytes(m.get("size")),
            "vram_gb": gb_of_bytes(m.get("size_vram")),
        }
        for m in (rows if running else None) or []
    ]
    return server, models


def ollama_rows(data: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The `models` of an `/api/ps` answer, `None` when it has none."""
    rows: Any = data.get("models")
    if not isinstance(rows, list):
        return None
    return [m for m in cast(list[Any], rows) if isinstance(m, dict)]


def comfyui_urls(registry: Registry) -> list[str]:
    """The distinct `base_url`s of the registry's `comfyui` entries, plus `HONE_COMFYUI_URL` when set."""
    env = os.environ.get("HONE_COMFYUI_URL")
    entries = sorted((c for c in registry.models.values() if provider(c) == "comfyui"), key=lambda c: c.id)
    urls = [(c.base_url or env or COMFYUI_URL).rstrip("/") for c in entries]
    return list(dict.fromkeys(urls + ([env.rstrip("/")] if env else [])))


def read_comfyui(url: str, registry: Registry, loaded: dict[str, Models]) -> tuple[Server, Models]:
    """The server's torch memory (`/system_stats`) and the models hone-models ran there since the last
    `/free`, plus an unnamed entry for a job or memory it cannot name."""
    running, stats, error = get_json(f"{url}/system_stats")
    server: Server = {"server": "comfyui", "url": url, "running": running, "error": error, "vram_gb": None}
    if not running:
        return server, []
    answered, history, error = get_json(f"{url}/history?max_items=1")
    if not answered:
        server.update(running=None, error=error or f"{url}/history refused the connection")
        return server, []
    server["vram_gb"] = vram = torch_vram_gb(stats)
    if vram == 0:  # no torch memory: restarted, or freed by someone else
        return server, []
    jobs = loaded.get(url, [])
    ids = list(dict.fromkeys(str(j["model_id"]) for j in jobs))
    models = [comfyui_entry(model_id, registry) for model_id in ids]
    newest = list(history)[-1] if history else None
    if (newest is not None and newest not in {j["job_id"] for j in jobs}) or (not ids and vram is not None):
        models.append(comfyui_entry(None, registry))
    return server, models


def torch_vram_gb(stats: dict[str, Any]) -> float | None:
    """The sum of the devices' `torch_vram_total` in GB; `None` when a device does not report it."""
    devices: Any = stats.get("devices")
    if not isinstance(devices, list) or not devices:
        return None
    totals = [
        cast(dict[str, Any], d).get("torch_vram_total") if isinstance(d, dict) else None
        for d in cast(list[Any], devices)
    ]
    if not all(isinstance(t, int | float) for t in totals):
        return None
    return gb_of_bytes(sum(cast(list[float], totals)))


def comfyui_entry(model_id: str | None, registry: Registry) -> dict[str, Any]:
    """A ComfyUI `loaded_models` entry, named by the entry's workflow file; sizes are per server only."""
    cfg = registry.models.get(model_id or "")
    workflow = getattr(cfg, "workflow", None)
    name = Path(str(workflow)).name if workflow else model_id
    return {"server": "comfyui", "name": name, "model_id": model_id, "size_gb": None, "vram_gb": None}
