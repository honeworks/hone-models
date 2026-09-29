"""Which registry models a ComfyUI server has run since it last freed its memory (change 0015 §2).

ComfyUI cannot name the models it holds, so hone-models keeps `${HONE_HOME}/models/comfyui-loaded.json`,
guarded by `flock` like the GPU ledger:

    {"http://127.0.0.1:8188": [{"model_id": "z-image-turbo", "job_id": "…", "pid": 4242, "time": "…"}]}

A job adds its model (`add`); every `/free` hone-models sends clears the server's list (`clear`). The
machine state (change 0016) reads it with `read`.
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


def path() -> Path:
    """The file (resolved on every use, so a test's `HONE_HOME` applies)."""
    return Path(os.environ.get("HONE_HOME", ".hone")) / "models" / "comfyui-loaded.json"


@contextmanager
def _locked() -> Generator[dict[str, list[dict[str, Any]]]]:
    """The file's content under an exclusive lock; changes are written back atomically."""
    file = path()
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            try:
                data: dict[str, list[dict[str, Any]]] = json.loads(file.read_text(encoding="utf-8"))
            except (FileNotFoundError, ValueError):
                data = {}
            yield data
            tmp = file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            tmp.replace(file)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def add(url: str, model_id: str, job_id: str) -> None:
    """Note that the server at `url` ran `model_id` in job `job_id` (this process, now)."""
    with _locked() as data:
        entry = {"model_id": model_id, "job_id": job_id, "pid": os.getpid(), "time": now_iso()}
        data.setdefault(url.rstrip("/"), []).append(entry)


def clear(url: str) -> None:
    """Forget what the server at `url` holds (it was just freed)."""
    with _locked() as data:
        data.pop(url.rstrip("/"), None)


def read() -> dict[str, list[dict[str, Any]]]:
    """Per server URL, the models run since its last `/free`, oldest first."""
    with _locked() as data:
        return {url: [dict(e) for e in entries] for url, entries in data.items()}
