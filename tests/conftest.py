"""Shared pytest fixtures: isolation, GPU lock and real-model availability checks."""

from __future__ import annotations

import fcntl
import json
import os
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # no network fetch when LiteLLM is imported
OLLAMA_URL = os.environ.get("HONE_TEST_OLLAMA_URL", "http://127.0.0.1:11434")
FIXTURES = Path(__file__).parent / "fixtures" / "http"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def recorded():
    """recorded("ollama_chat_ok.json") -> the recorded HTTP response body from tests/fixtures/http/."""
    return load_fixture


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries without sleeping (unit tests)."""
    monkeypatch.setattr("hone_models._http.BACKOFF_S", 0.0)


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own HONE_HOME, HOME (no user registry) and working directory."""
    monkeypatch.setenv("HONE_HOME", str(tmp_path / ".hone"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HONE_CAPTURE_CONTENT", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hone_models.records._SECRETS", set())
    monkeypatch.setattr("hone_models.providers.ollama.loaded", set())  # a short lease would unload these
    return tmp_path


@pytest.fixture(scope="session")
def gpu_lock() -> Iterator[None]:
    """Hold the machine-wide GPU lock for the session (no-op if scripts/gpu-lock.sh already holds it)."""
    if os.environ.get("HONE_GPU_LOCK_HELD") == "1":
        yield
        return
    path = Path(os.environ.get("HONE_GPU_LOCK", "/tmp/honeworks-gpu.lock"))  # noqa: S108 - shared machine-wide lock by design
    path.touch(exist_ok=True)
    with path.open("r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:  # noqa: S310
            return [m["name"] for m in json.load(r).get("models", [])]
    except OSError:
        return []


@pytest.fixture(scope="session")
def ollama_model(gpu_lock: None):
    """Factory: ollama_model("HONE_TEST_TEXT_MODEL", "gemma4-12b:latest") -> name, or skip with a reason."""
    available = ollama_models()

    loaded: list[str] = []

    def _use(env: str, default: str) -> str:
        name = os.environ.get(env, default)
        if not available:
            pytest.skip(f"Ollama not reachable at {OLLAMA_URL}")
        if name not in available:
            pytest.skip(f"Ollama model {name!r} ({env}) not installed")
        loaded.append(name)
        return name

    yield _use
    for name in set(loaded):  # free the shared GPU for the next test run
        try:
            req = urllib.request.Request(  # noqa: S310
                f"{OLLAMA_URL}/api/generate",
                data=json.dumps({"model": name, "keep_alive": 0}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=30).read()  # noqa: S310
        except OSError:
            pass
