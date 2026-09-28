"""AC-11: a transient 429 then success is retried (recorded as span events); a 400 is not retried."""

import httpx
import pytest
import respx

import hone_models as mk

pytestmark = pytest.mark.e2e
BASE = "http://localhost:8080/v1"
OK = {"model": "local", "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {}}


@pytest.fixture
def llm(isolated) -> mk.TextClient:
    path = isolated / "reg.toml"
    path.write_text(f'[models.local]\nprovider = "openai_compatible"\nbase_url = "{BASE}"\n')
    return mk.text("local", registry=mk.registry.load([path]), sink=mk.records.MemorySink())


def test_ac11_transient_429_then_success(llm) -> None:
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/chat/completions")
        route.side_effect = [respx.MockResponse(429, text="slow down"), respx.MockResponse(json=OK)]
        r = llm.complete([{"role": "user", "content": "hello"}])
    assert r.text == "hi"
    assert route.call_count == 2
    span = llm.sink.spans[0]
    assert span["status"]["code"] == "ok"
    assert [(e["name"], e["attributes"]) for e in span["events"]] == [
        ("retry", {"attempt": 2, "reason": "http 429"})
    ]


def test_ac11_400_is_not_retried(llm) -> None:
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/chat/completions").respond(400, json={"error": "image input not supported"})
        with pytest.raises(mk.errors.ProviderError, match="HTTP 400") as info:
            llm.complete([{"role": "user", "content": "hello"}])
    assert info.value.status == 400
    assert route.call_count == 1
    span = llm.sink.spans[0]
    assert span["events"] == []
    assert span["status"]["code"] == "error"


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Backoff sleeps, recorded instead of slept."""
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    return slept


def test_ac11_gives_up_after_three_attempts(llm, sleeps) -> None:
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/chat/completions").respond(503)
        with pytest.raises(mk.errors.ProviderError, match="failed after 3 attempts"):
            llm.complete([{"role": "user", "content": "hello"}])
    assert route.call_count == 3
    assert len(sleeps) == 2
    span = llm.sink.spans[0]
    assert [e["attributes"]["attempt"] for e in span["events"]] == [2, 3]
    assert span["status"]["code"] == "error"


def test_ac11_connect_timeout_is_retried_read_timeout_is_not(llm, sleeps) -> None:
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/chat/completions")
        route.side_effect = [httpx.ConnectTimeout("no route"), respx.MockResponse(json=OK)]
        assert llm.complete([{"role": "user", "content": "hello"}]).text == "hi"
        assert route.call_count == 2
        route.side_effect = [httpx.ReadTimeout("slow")]
        with pytest.raises(mk.errors.ModelTimeout):
            llm.complete([{"role": "user", "content": "hello"}])
    assert route.call_count == 3
    assert llm.sink.spans[1]["events"] == []
