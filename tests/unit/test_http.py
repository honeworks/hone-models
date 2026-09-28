import httpx
import pytest
import respx

from hone_models._http import api_key, request_json
from hone_models._tracing import start_span
from hone_models.budget import timeout_for
from hone_models.errors import ConfigError, ModelTimeout, ProviderError
from hone_models.records import MemorySink, prepare
from hone_models.registry import Capabilities, ModelConfig

URL = "http://h/x"


@respx.mock
def test_transient_errors_retry_and_are_recorded(no_backoff) -> None:
    route = respx.post(URL)
    route.side_effect = [httpx.Response(500), httpx.Response(429), httpx.Response(200, json={"ok": 1})]
    sink = MemorySink()
    with start_span("s", sink):
        assert request_json("POST", URL, payload={}, timeout=5) == {"ok": 1}
    events = sink.spans[0]["events"]
    assert [(e["name"], e["attributes"]["attempt"], e["attributes"]["reason"]) for e in events] == [
        ("retry", 2, "http 500"),
        ("retry", 3, "http 429"),
    ]


@respx.mock
def test_client_errors_do_not_retry() -> None:
    route = respx.post(URL).respond(404, text="model not found")
    with pytest.raises(ProviderError, match="HTTP 404: model not found") as info:
        request_json("POST", URL, timeout=5)
    assert info.value.status == 404
    assert route.call_count == 1


@respx.mock
def test_gives_up_after_three_attempts(no_backoff) -> None:
    route = respx.get(URL).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ProviderError, match=r"failed after 3 attempts \(ConnectError: refused\)"):
        request_json("GET", URL, timeout=5)
    assert route.call_count == 3


@respx.mock
def test_timeout_and_bad_json() -> None:
    respx.get(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(ModelTimeout, match="did not answer within 5s"):
        request_json("GET", URL, timeout=5)
    respx.get(URL).respond(200, text="<html>")
    with pytest.raises(ProviderError, match="invalid JSON"):
        request_json("GET", URL, timeout=5)
    respx.get(URL).mock(side_effect=httpx.UnsupportedProtocol("ftp"))
    with pytest.raises(ProviderError, match="request to http://h/x failed"):
        request_json("GET", URL, timeout=5)


def test_timeout_from_measured_speed() -> None:
    slow = ModelConfig(id="m", provider="ollama", capabilities=Capabilities(speed_tok_s=10.0))
    assert timeout_for(slow, 1000) == 200.0
    assert timeout_for(slow, 100) == 120.0  # minimum
    assert timeout_for(slow, 100_000) == 600.0  # config max


def test_timeout_without_measured_speed_grows_for_local_models_only() -> None:
    """Change 0010: concept-shorts' max_tokens=7000 script call timed out after 120 s on a fresh machine."""
    local = ModelConfig(id="m", provider="ollama")
    assert timeout_for(local, 2000) == 400.0  # assumed 10 tokens/s
    assert timeout_for(local, 7000) == 600.0  # still capped by max_timeout_s
    assert timeout_for(local, 100) == 120.0
    hosted = ModelConfig(id="h", provider="openai_compatible", base_url="https://api.example.com/v1")
    assert timeout_for(hosted, 8000) == 120.0  # unknown hosted speed: the minimum, as before


def test_api_key_is_registered_as_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ModelConfig(id="m", provider="openai_compatible", api_key_env="MY_KEY")
    with pytest.raises(ConfigError, match="set the environment variable MY_KEY"):
        api_key(cfg)
    monkeypatch.setenv("MY_KEY", "key-value-123456")
    assert api_key(cfg) == "key-value-123456"
    assert prepare({"attributes": {"a": "key-value-123456"}}, True)["attributes"]["a"] == "***"
    assert api_key(ModelConfig(id="m", provider="ollama")) is None


@respx.mock
def test_backoff_is_exponential_with_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    monkeypatch.setattr("random.uniform", lambda a, b: 0.75)
    respx.post(URL).respond(500)
    with pytest.raises(ProviderError):
        request_json("POST", URL, payload={}, timeout=5)
    assert slept == [0.5 * 0.75, 1.0 * 0.75]
