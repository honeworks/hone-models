"""Reading GPU memory for the lease (design/current.md §7): the whole card and per process.

NVML (extra `gpu`) is preferred; `nvidia-smi` is the fallback; `None` means unknown.
"""

from __future__ import annotations

import importlib
import subprocess
from collections.abc import Callable
from typing import Any

MB = 1024 * 1024

MemoryReader = Callable[[], "tuple[int, int] | None"]  # (total MB, used MB), None when unknown
ProcessReader = Callable[[], "dict[int, int] | None"]  # {pid: used MB}, None when unknown


def read_memory() -> tuple[int, int] | None:
    """(total MB, used MB) of GPU 0 from NVML, else from `nvidia-smi`, else `None`."""

    def query(nvml: Any, handle: Any) -> tuple[int, int]:
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.total) // MB, int(info.used) // MB

    found = _nvml(query)
    return _smi_memory() if found is None else found


def read_processes() -> dict[int, int] | None:
    """GPU memory used per process on GPU 0 (`{pid: MB}`) from NVML, else `nvidia-smi`, else `None`."""

    def query(nvml: Any, handle: Any) -> dict[int, int]:
        procs = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
        return {int(p.pid): int(p.usedGpuMemory or 0) // MB for p in procs}

    found = _nvml(query)
    return _smi_processes() if found is None else found


def _nvml(query: Callable[[Any, Any], Any]) -> Any:
    try:
        nvml: Any = importlib.import_module("pynvml")
    except ImportError:
        return None
    try:
        nvml.nvmlInit()
        try:
            return query(nvml, nvml.nvmlDeviceGetHandleByIndex(0))
        finally:
            nvml.nvmlShutdown()
    except (nvml.NVMLError, AttributeError):  # AttributeError: an old or partial binding
        return None


def _smi(fields: str, what: str = "gpu") -> list[list[int]] | None:
    query = ["nvidia-smi", f"--query-{what}={fields}", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(query, capture_output=True, text=True, timeout=10, check=True).stdout  # noqa: S603
        return [[int(x) for x in line.split(",")] for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _smi_memory() -> tuple[int, int] | None:
    rows = _smi("memory.total,memory.used")
    return (rows[0][0], rows[0][1]) if rows else None


def _smi_processes() -> dict[int, int] | None:
    rows = _smi("pid,used_memory", "compute-apps")
    return None if rows is None else {row[0]: row[1] for row in rows}


def free_mb(memory: tuple[int, int] | None, entries: list[dict[str, Any]]) -> int | None:
    """`total - max(used, reserved)`, or `None` when GPU memory is unknown."""
    if memory is None:
        return None
    total, used = memory
    return total - max(used, sum(int(e["vram_mb"]) for e in entries))
