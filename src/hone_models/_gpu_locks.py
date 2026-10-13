"""The two simple GPU lease shapes (design/current.md §7): one that never waits, one exclusive lock; a
look at the machine-wide lock of `scripts/gpu-lock.sh` (change 0016); and the caller's cancel event that
ends a lease's wait (change 0020 §4, D-084)."""

from __future__ import annotations

import contextlib
import fcntl
import os
import threading
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from typing import IO, Any

from .errors import Cancelled

MACHINE_LOCK = "/tmp/honeworks-gpu.lock"  # noqa: S108 - the machine-wide lock of scripts/gpu-lock.sh
_cancel: ContextVar[threading.Event | None] = ContextVar("hone_models_lease_cancel", default=None)


@contextmanager
def cancel_waits(event: threading.Event | None) -> Generator[None]:
    """Let `event` end the lease waits started in this block (`Cancelled`); the lease port is unchanged."""
    token = _cancel.set(event)
    try:
        yield
    finally:
        _cancel.reset(token)


def pause(seconds: float, name: str) -> None:
    """Sleep `seconds` of a lease's wait, or raise `Cancelled` as soon as the caller's event is set."""
    event = _cancel.get()
    if event is None:
        time.sleep(seconds)
    elif event.wait(seconds):
        raise Cancelled(f"the wait for the GPU lease {name!r} was cancelled")


def machine_lock_path() -> Path:
    """The machine-wide GPU lock file: `HONE_GPU_LOCK`, else `/tmp/honeworks-gpu.lock` (as gpu-lock.sh)."""
    return Path(os.environ.get("HONE_GPU_LOCK") or MACHINE_LOCK)


def probe_lock(path: Path) -> dict[str, Any]:
    """`{"path", "held", "mine", "holder"}` for the lock file at `path`, without keeping it.

    `held` comes from a non-blocking `flock` released at once (`None` when the file cannot be opened; a
    missing file is not held); `mine` from `HONE_GPU_LOCK_HELD=1` (our own gpu-lock.sh holds it); `holder`
    from `<path>.holder`, read only when held because a killed run can leave it behind."""
    state: dict[str, Any] = {"path": str(path), "held": False, "mine": False, "holder": None}
    try:
        with path.open("rb") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh, fcntl.LOCK_UN)
                return state
            except BlockingIOError:
                state["held"] = True
    except FileNotFoundError:
        return state
    except OSError:
        return {**state, "held": None, "mine": None}
    state["mine"] = os.environ.get("HONE_GPU_LOCK_HELD") == "1"
    with contextlib.suppress(OSError):  # no holder file: the holder is unknown
        state["holder"] = Path(f"{path}.holder").read_text(encoding="utf-8").strip() or None
    return state


class NullGpuLease:
    """A lease that never waits (the default in consumers without a GPU)."""

    def lease(
        self,
        name: str,
        vram_gb: float,
        *,
        timeout_s: float | None = None,
        trace: Mapping[str, str] | None = None,
    ) -> nullcontext[None]:
        return nullcontext()


class FileLockGpuLease:
    """One GPU user at a time on this machine: an exclusive `flock` on `path`, reentrant per thread."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._local = threading.local()

    @contextmanager
    def lease(
        self,
        name: str,
        vram_gb: float,
        *,
        timeout_s: float | None = None,
        trace: Mapping[str, str] | None = None,
    ) -> Generator[None]:
        if getattr(self._local, "held", False):
            yield
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            _flock(fh, name, timeout_s)
            self._local.held = True
            try:
                yield
            finally:
                self._local.held = False
                fcntl.flock(fh, fcntl.LOCK_UN)


def _flock(fh: IO[str], name: str, timeout_s: float | None) -> None:
    if timeout_s is None and _cancel.get() is None:
        fcntl.flock(fh, fcntl.LOCK_EX)
        return
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"GPU lock {fh.name} for {name!r} not free within {timeout_s} s") from None
            pause(0.1, name)
