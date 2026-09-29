"""The ComfyUI models hone-models ran since the last `/free`, per server URL (changes 0015 §2, 0016).

ComfyUI cannot name the models it holds, so every job adds its registry id here and every `/free`
hone-models sends clears the server's list. The file is `${HONE_HOME}/models/comfyui-loaded.json`,
`{url: [{"model_id", "job_id", "pid", "time"}]}`, guarded by `flock` like the lease ledger.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .records import now_iso


def _path() -> Path:
    return Path(os.environ.get("HONE_HOME", ".hone")) / "models" / "comfyui-loaded.json"


@contextmanager
def _locked() -> Generator[dict[str, list[dict[str, Any]]]]:
    """The file's content under an exclusive lock; changes are written back atomically."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            try:
                data: dict[str, list[dict[str, Any]]] = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, ValueError):
                data = {}
            yield data
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            tmp.replace(path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read() -> dict[str, list[dict[str, Any]]]:
    """`{url: [{"model_id", "job_id", "pid", "time"}]}`: the models run on each server since its `/free`."""
    with _locked() as data:
        return {url: list(entries) for url, entries in data.items()}


def add(url: str, model_id: str, job_id: str) -> None:
    """Note that `model_id` ran as job `job_id` on the server at `url`."""
    entry = {"model_id": model_id, "job_id": job_id, "pid": os.getpid(), "time": now_iso()}
    with _locked() as data:
        data.setdefault(url.rstrip("/"), []).append(entry)


def clear(url: str) -> None:
    """Forget the server's models (after a `/free`)."""
    with _locked() as data:
        data.pop(url.rstrip("/"), None)
