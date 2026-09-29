"""Reading GPU memory for the lease (design/current.md §7): the whole card and per process; and every GPU
with its name and utilization for `mk.machine.snapshot()` (change 0016).

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
# [{"index", "name", "memory_total_mb", "memory_used_mb", "utilization_pct"}], None when unknown
GpusReader = Callable[[], "list[dict[str, Any]] | None"]


def read_memory() -> tuple[int, int] | None:
    """(total MB, used MB) of GPU 0 from NVML, else from `nvidia-smi`, else `None`."""

    def query(nvml: Any) -> tuple[int, int]:
        info = nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(0))
        return int(info.total) // MB, int(info.used) // MB

    found = _nvml(query)
    return _smi_memory() if found is None else found


def read_processes() -> dict[int, int] | None:
    """GPU memory used per process on GPU 0 (`{pid: MB}`) from NVML, else `nvidia-smi`, else `None`."""

    def query(nvml: Any) -> dict[int, int]:
        procs = nvml.nvmlDeviceGetComputeRunningProcesses(nvml.nvmlDeviceGetHandleByIndex(0))
        return {int(p.pid): int(p.usedGpuMemory or 0) // MB for p in procs}

    found = _nvml(query)
    return _smi_processes() if found is None else found


def read_gpus() -> list[dict[str, Any]] | None:
    """Every GPU with its name, memory (MB) and utilization (%, `None` when the reader cannot tell), from
    NVML, else `nvidia-smi`, else `None`."""

    def query(nvml: Any) -> list[dict[str, Any]]:
        return [_nvml_gpu(nvml, index) for index in range(int(nvml.nvmlDeviceGetCount()))]

    found = _nvml(query)
    return _smi_gpus() if found is None else found


def _nvml_gpu(nvml: Any, index: int) -> dict[str, Any]:
    handle = nvml.nvmlDeviceGetHandleByIndex(index)
    name = nvml.nvmlDeviceGetName(handle)
    info = nvml.nvmlDeviceGetMemoryInfo(handle)
    try:
        utilization: int | None = int(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except nvml.NVMLError:  # not supported on this device
        utilization = None
    return {
        "index": index,
        "name": name.decode() if isinstance(name, bytes) else str(name),
        "memory_total_mb": int(info.total) // MB,
        "memory_used_mb": int(info.used) // MB,
        "utilization_pct": utilization,
    }


def _nvml(query: Callable[[Any], Any]) -> Any:
    try:
        nvml: Any = importlib.import_module("pynvml")
    except ImportError:
        return None
    try:
        nvml.nvmlInit()
        try:
            return query(nvml)
        finally:
            nvml.nvmlShutdown()
    except (nvml.NVMLError, AttributeError):  # AttributeError: an old or partial binding
        return None


def _smi_rows(fields: str, what: str = "gpu") -> list[list[str]] | None:
    query = ["nvidia-smi", f"--query-{what}={fields}", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(query, capture_output=True, text=True, timeout=10, check=True).stdout  # noqa: S603
    except (OSError, subprocess.SubprocessError):
        return None
    return [[x.strip() for x in line.split(",")] for line in out.splitlines() if line.strip()]


def _smi(fields: str, what: str = "gpu") -> list[list[int]] | None:
    rows = _smi_rows(fields, what)
    try:
        return None if rows is None else [[int(x) for x in row] for row in rows]
    except ValueError:
        return None


def _smi_gpus() -> list[dict[str, Any]] | None:
    rows = _smi_rows("index,name,memory.total,memory.used,utilization.gpu")
    try:
        return [
            {
                "index": int(index),
                "name": name,
                "memory_total_mb": int(total),
                "memory_used_mb": int(used),
                "utilization_pct": int(utilization) if utilization.isdigit() else None,  # "[N/A]"
            }
            for index, name, total, used, utilization in rows or []
        ] or None
    except ValueError:  # an unexpected format: unknown, not a crash
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
