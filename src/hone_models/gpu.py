"""GPU scheduling on one machine (design/current.md §7), with the GPU lease shape of §10.

    with mk.gpu.lease("whisper-turbo", vram_gb=6):
        run_whisper()                    # non-LLM GPU work takes part in scheduling
    mk.gpu.status()                      # total / used / free MB, the leases held and those waiting

A lease reserves memory in a JSON ledger (`${HONE_HOME}/models/gpu-ledger.json`) shared by every process
and guarded by `fcntl.flock`; entries of dead processes are dropped. Free memory is
`total - max(used, reserved)`: used memory comes from NVML (extra `gpu`, fallback `nvidia-smi`), reserved
memory from the ledger, so a lease counts before its model is loaded and other programs count too. Memory
the leasing process itself uses does not count against it (change 0004).
When memory is short, Ollama models this process loaded are unloaded once (all running models with
`unload_others=True`; with `if_busy="block"` only while no other process holds a lease or the machine-wide
lock) and the release hooks registered with `on_short` are called, then the lease waits,
polling every 0.5 s, until `timeout_s` (`TimeoutError`).
A wait that cannot end (no other lease held and free memory unchanged for `stall_s`) raises
`CapabilityError`; a long wait is logged and listed in `status().waiting`. A lease taken inside another
one in the same thread is part of it: it reserves only what the outer leases do not cover.
Without any GPU information, leases are granted at once. Spans made inside a lease carry
`hone.models.gpu.*` attributes.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import secrets
import threading
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._gpu_locks import FileLockGpuLease, NullGpuLease
from ._gpu_memory import MemoryReader, ProcessReader, free_mb, read_memory, read_processes
from ._gpu_room import (
    IfBusy,
    busy_elsewhere,
    check_if_busy,
    comfyui_free,
    on_short,
    run_hooks,
    stalled,
    torch_empty_cache,
    unload_idle,
)
from ._tracing import extra_attributes
from .errors import CapabilityError
from .records import log, now_iso

__all__ = [
    "GPU",
    "FileLockGpuLease",
    "GpuScheduler",
    "GpuStatus",
    "NullGpuLease",
    "comfyui_free",
    "free_mb",
    "lease",
    "on_short",
    "read_memory",
    "read_processes",
    "status",
    "torch_empty_cache",
]

POLL_S = 0.5
STALL_S = 120.0  # free memory unchanged this long with no other lease held: the lease can never be granted
WARN_AFTER_S = 30.0  # a waiting lease is logged after this long, then every WARN_EVERY_S
WARN_EVERY_S = 300.0


@dataclass(frozen=True, slots=True)
class GpuStatus:
    """GPU memory in MB (`None` when unknown), the leases held on this machine and the leases waiting."""

    total_mb: int | None
    used_mb: int | None
    free_mb: int | None
    leases: list[dict[str, Any]]
    waiting: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])


@dataclass(frozen=True, slots=True)
class _Held:
    """A lease this thread holds; `mb` is what it reserved on top of the leases around it."""

    ledger: Path
    pid: int
    name: str
    mb: int
    entry_id: str | None


_THREAD = threading.local()  # per thread: the leases held, outermost first


def _held() -> list[_Held]:
    if not hasattr(_THREAD, "stack"):
        _THREAD.stack = list[_Held]()
    stack: list[_Held] = _THREAD.stack
    return stack


def _valid(entry: Any) -> bool:
    return isinstance(entry, dict) and {"id", "pid", "vram_mb"} <= entry.keys()  # pyright: ignore[reportUnknownMemberType]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


class GpuScheduler:
    """Leases GPU memory across processes through the shared ledger (the default is `mk.gpu.GPU`).

    `memory` reads (total MB, used MB) and `processes` reads {pid: used MB}; both return `None` when
    unknown. `stall_s` is how long a wait with no other lease held and no change may last. With
    `unload_others`, `if_busy="block"` leaves other processes' models alone while another process holds a
    lease or the machine-wide GPU lock; `"unload"` (the default) unloads them anyway (change 0016)."""

    def __init__(
        self,
        ledger: str | Path | None = None,
        *,
        memory: MemoryReader = read_memory,
        unload_others: bool = False,
        processes: ProcessReader = read_processes,
        stall_s: float = STALL_S,
        if_busy: IfBusy = "unload",
    ) -> None:
        self._ledger = Path(ledger) if ledger else None
        self.memory = memory
        self.processes = processes
        self.unload_others = unload_others
        self.stall_s = stall_s
        self.if_busy = check_if_busy(if_busy)

    @property
    def ledger_path(self) -> Path:
        """The ledger file (default `${HONE_HOME:-.hone}/models/gpu-ledger.json`)."""
        # Resolved on every use: `GPU` is created at import, before callers (and tests) set HONE_HOME.
        return self._ledger or Path(os.environ.get("HONE_HOME", ".hone")) / "models" / "gpu-ledger.json"

    @contextmanager
    def _entries(self, path: Path | None = None) -> Generator[list[dict[str, Any]]]:
        """The live entries of the ledger (or of `path` next to it) under an exclusive lock; changes are
        written back atomically."""
        path = path or self.ledger_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                try:
                    entries: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
                except (FileNotFoundError, ValueError):
                    entries = []
                entries = [e for e in entries if _valid(e) and _alive(int(e["pid"]))]
                yield entries
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(entries, indent=1), encoding="utf-8")
                tmp.replace(path)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @property
    def _waiting_path(self) -> Path:
        return self.ledger_path.with_name("gpu-waiting.json")

    def status(self) -> GpuStatus:
        """Total / used / free MB (`None` without GPU information), the leases held and those waiting."""
        with self._entries() as entries:
            leases = [dict(e) for e in entries]
        with self._entries(self._waiting_path) as entries:
            waiting = [dict(e) for e in entries]
        memory = self.memory()
        total, used = memory if memory else (None, None)
        return GpuStatus(total, used, free_mb(memory, leases), leases, waiting)

    @contextmanager
    def lease(
        self,
        name: str,
        vram_gb: float,
        *,
        timeout_s: float | None = None,
        trace: Mapping[str, str] | None = None,  # part of the port; a lease records no span of its own
    ) -> Generator[None]:
        """Reserve `vram_gb` for the duration of the block. Reentrant for `name` in the same thread; a
        lease inside another one in the same thread reserves only what the outer leases do not cover."""
        ledger = self.ledger_path.resolve()
        stack = _held()
        outer = [h for h in stack if h.ledger == ledger and h.pid == os.getpid()]  # a forked child: none
        if any(h.name == name for h in outer):
            yield
            return
        need = math.ceil(vram_gb * 1024)
        extra = need - sum(h.mb for h in outer)
        entry_id, attrs = None, None
        if not outer or extra > 0:
            entry_id, attrs = self._acquire(name, need, extra, timeout_s, outer)
        held = _Held(ledger, os.getpid(), name, extra if entry_id else 0, entry_id)
        stack.append(held)
        try:
            with extra_attributes(attrs) if attrs else nullcontext():
                yield
        finally:
            stack.remove(held)
            if entry_id:
                with self._entries() as entries:
                    entries[:] = [e for e in entries if e["id"] != entry_id]

    def _acquire(
        self, name: str, need: int, extra: int, timeout_s: float | None, outer: list[_Held]
    ) -> tuple[str, dict[str, Any]]:
        entry: dict[str, Any] = {"id": secrets.token_hex(8), "name": name, "pid": os.getpid()}
        entry["vram_mb"] = extra
        if outer:
            entry["nested_in"] = outer[0].name
        mine = {h.entry_id for h in outer}
        wait = _Wait(self, entry, need, timeout_s)
        try:
            while True:
                with self._entries() as entries:
                    memory = self.memory()
                    free = self._free_for_me(memory, entries, extra)
                    if free is None or free >= extra:
                        entries.append({**entry, "since": now_iso()})
                        break
                    others = [e for e in entries if e["id"] not in mine]
                if memory is not None and memory[0] < need:
                    total = memory[0]
                    raise CapabilityError(f"GPU lease {name!r} needs {need} MB but the GPU has {total} MB")
                wait.step(free, others)
        finally:
            wait.done()
        attrs: dict[str, Any] = {
            "hone.models.gpu.lease_wait_ms": round(wait.waited() * 1000),
            "hone.models.gpu.unloaded": wait.unloaded or [],
            "hone.models.gpu.released": wait.released,
        }
        if memory is not None:
            attrs["hone.models.gpu.vram_before_mb"] = memory[1]
        return entry["id"], attrs

    def _free_for_me(
        self, memory: tuple[int, int] | None, entries: list[dict[str, Any]], want: int
    ) -> int | None:
        """Free memory, not counting what this process itself uses as taken (its CUDA context or cached
        torch memory is no reason to wait for itself)."""
        free = free_mb(memory, entries)
        if memory is None or free is None or free >= want:
            return free
        own = (self.processes() or {}).get(os.getpid())
        return free_mb((memory[0], max(0, memory[1] - own)), entries) if own else free


class _Wait:
    """One lease's wait: the single attempt to make room, the timeout, the stall clock, the log and the
    `waiting` list."""

    def __init__(self, scheduler: GpuScheduler, entry: dict[str, Any], need: int, timeout_s: float | None):
        self.gpu, self.entry, self.need, self.timeout_s = scheduler, entry, need, timeout_s
        self.start = self.stall_start = time.monotonic()
        self.last_free: int | None = None
        self.unloaded: list[str] | None = None
        self.released: list[str] = []
        self.warn_at = WARN_AFTER_S
        self.listed = False

    def waited(self) -> float:
        return time.monotonic() - self.start

    def step(self, free: int, others: list[dict[str, Any]]) -> None:
        """Make room once, else raise on timeout or stall, else log and sleep one poll."""
        waited = self.waited()
        if self.unloaded is None and (self.timeout_s is None or waited < self.timeout_s):
            gpu = self.gpu
            yield_to_others = gpu.if_busy == "block" and gpu.unload_others and busy_elsewhere(others)
            self.unloaded = unload_idle(gpu.unload_others and not yield_to_others)
            self.released = run_hooks()
            return
        if self.timeout_s is not None and waited >= self.timeout_s:
            raise TimeoutError(
                f"GPU lease {self.entry['name']!r} ({self.need} MB) not granted within {self.timeout_s} s: "
                f"{free} MB free; see mk.gpu.status() for the leases held"
            )
        now = time.monotonic()
        if others or free != self.last_free:
            self.stall_start, self.last_free = now, free
        elif now - self.stall_start >= self.gpu.stall_s:
            gpu = self.gpu
            procs = gpu.processes()
            raise stalled(
                self.entry["name"],
                self.entry["vram_mb"],  # what it adds to the ledger: less than `need` when nested
                free,
                stall_s=gpu.stall_s,
                processes=procs,
                unload_others=gpu.unload_others,
                nested_in=self.entry.get("nested_in"),
            )
        self._report(waited, free, others)
        time.sleep(POLL_S)

    def _report(self, waited: float, free: int, others: list[dict[str, Any]]) -> None:
        if not self.listed:
            with self.gpu._entries(self.gpu._waiting_path) as waiting:  # pyright: ignore[reportPrivateUsage]
                waiting.append({**self.entry, "vram_mb": self.need, "since": now_iso()})
            self.listed = True
        if waited >= self.warn_at:
            self.warn_at += WARN_EVERY_S
            held = [f"{e.get('name')} ({e['vram_mb']} MB, pid {e['pid']})" for e in others] or "none"
            log.warning(
                "hone-models: GPU lease %r (%d MB) has waited %.0f s: %d MB free; leases held: %s",
                self.entry["name"], self.need, waited, free, held,
            )  # fmt: skip

    def done(self) -> None:
        if self.listed:
            with self.gpu._entries(self.gpu._waiting_path) as waiting:  # pyright: ignore[reportPrivateUsage]
                waiting[:] = [e for e in waiting if e["id"] != self.entry["id"]]


GPU = GpuScheduler()
lease = GPU.lease
status = GPU.status
