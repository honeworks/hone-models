import json

import pytest
import respx

import hone_models as mk
from hone_models.errors import ConfigError
from hone_models.providers import ollama

URL = "http://127.0.0.1:11434"


def test_unload_sends_keep_alive_zero_and_forgets_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "loaded", {(URL, "gemma4-12b:latest")})
    with respx.mock(base_url=URL) as mock:
        route = mock.post("/api/generate").respond(json={"done": True})
        mk.unload("gemma4-12b")
    assert json.loads(route.calls.last.request.content) == {"model": "gemma4-12b:latest", "keep_alive": 0}
    assert ollama.loaded == set()


def test_unload_uses_the_model_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box:11434")
    with respx.mock() as mock:
        route = mock.post("http://gpu-box:11434/api/generate").respond(json={})
        mk.unload("ollama:llama3.2:1b")
    assert json.loads(route.calls.last.request.content)["model"] == "llama3.2:1b"


def test_hosted_models_cannot_be_unloaded() -> None:
    with pytest.raises(ConfigError, match="only Ollama and ComfyUI models"):
        mk.unload("gpt-4.1-mini")
