"""Helpers for the real ComfyUI tests: point hone-models at the real server and ComfyUI folder (the
autouse fixture moved both away), skip with a reason when neither a running server nor a start command
is there, and read the GPU memory in use."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

from hone_models.providers import _comfyui_server

# Read at import, before the autouse fixture points HOME at a temporary folder.
URL = os.environ.get("HONE_COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFYUI_DIR = os.environ.get("HONE_COMFYUI_DIR") or str(Path.home() / "ComfyUI")
START = os.environ.get("HONE_COMFYUI_START", "")


def use_real_comfyui(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Point hone-models at the real ComfyUI; return whether a server was already running there. Skip
    when none runs and `HONE_COMFYUI_START` is not set."""
    monkeypatch.setenv("HONE_COMFYUI_URL", URL)
    monkeypatch.setenv("HONE_COMFYUI_DIR", COMFYUI_DIR)
    running = _comfyui_server.healthy(URL)
    if not running and not START:
        pytest.skip(f"no ComfyUI at {URL} and HONE_COMFYUI_START is not set")
    if START:
        monkeypatch.setenv("HONE_COMFYUI_START", START)
    return running


def gpu_used_mb() -> int | None:
    """Memory in use on GPU 0 in MB (NVML, extra `gpu`), or `None` without NVML."""
    if importlib.util.find_spec("pynvml") is None:
        return None
    nvml: Any = importlib.import_module("pynvml")
    nvml.nvmlInit()
    try:
        return int(nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(0)).used) // 2**20
    finally:
        nvml.nvmlShutdown()
