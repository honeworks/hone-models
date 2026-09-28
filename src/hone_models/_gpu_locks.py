"""The two simple GPU lease shapes (design/current.md §7): one that never waits, one exclusive lock."""

from __future__ import annotations

import fcntl
import threading
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import IO


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
    if timeout_s is None:
        fcntl.flock(fh, fcntl.LOCK_EX)
        return
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"GPU lock {fh.name} for {name!r} not free within {timeout_s} s") from None
            time.sleep(0.1)
