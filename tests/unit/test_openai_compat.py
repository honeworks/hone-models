import json
import math

import pytest
import respx

import hone_models as mk
from hone_models.errors import ConfigError, ProviderError

BASE = "https://api.openai.com/v1"


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0000000000000000000")
    return "sk-test-0000000000000000000"


def test_chat_payload_schema_logprobs_and_images(key, recorded) -> None:
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    sink = mk.records.MemorySink()
    llm = mk.text("gpt-4.1-mini", sink=sink)
    image = {"type": "image", "data_b64": "QUJD", "mime": "image/png"}
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/chat/completions").respond(json=recorded("openai_chat_logprobs.json"))
        r = llm.complete(
            [{"role": "user", "content": [{"type": "text", "text": "ok?"}, image]}],
            schema=schema,
            logprobs=True,
            seed=3,
        )
    req = route.calls.last.request
    assert req.headers["Authorization"] == f"Bearer {key}"
    body = json.loads(req.content)
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "output", "schema": schema},
    }
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 5
    assert body["seed"] == 3
    assert body["messages"][-1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QUJD"},
    }
    assert r.parsed == {"ok": True}
    assert r.structured_path == "constrained"
    assert r.usage == {"input_tokens": 42, "output_tokens": 5}
    assert r.model == "gpt-4.1-mini-2025-04-14"
    assert r.logprobs is not None
    yes_step = r.logprobs[3]
    assert yes_step["token"] == " true"
    assert math.isclose(yes_step["top"][" false"], -1.6)
    span = sink.spans[0]
    assert span["attributes"]["gen_ai.provider.name"] == "openai"
    assert span["attributes"]["hone.models.model_digest"] == "fp_abc123"
    assert span["attributes"]["hone.models.cost_usd"] == pytest.approx((42 * 0.4 + 5 * 1.6) / 1e6)


def test_missing_key_base_url_and_bad_reply(isolated, key, monkeypatch) -> None:
    path = isolated / "r.toml"
    path.write_text('[models.nourl]\nprovider = "openai_compatible"\n')
    llm = mk.text("nourl", registry=mk.registry.load([path]), sink=mk.records.NullSink())
    with pytest.raises(ConfigError, match="needs base_url"):
        llm.complete([{"role": "user", "content": "x"}])
    with respx.mock(base_url=BASE) as mock:
        mock.post("/chat/completions").respond(json={"choices": []})
        with pytest.raises(ProviderError, match="no choices"):
            mk.text("gpt-4.1-mini", sink=mk.records.NullSink()).complete([{"role": "user", "content": "x"}])
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        mk.text("openai:gpt-4.1", sink=mk.records.NullSink()).complete([{"role": "user", "content": "x"}])


def test_reasoning_content_counts_as_thinking(key) -> None:
    reply = {"choices": [{"message": {"content": "", "reasoning_content": "hmm"}, "finish_reason": "length"}]}
    with respx.mock(base_url=BASE) as mock:
        mock.post("/chat/completions").respond(json=reply)
        r = mk.text("gpt-4.1-mini", sink=mk.records.NullSink()).complete([{"role": "user", "content": "x"}])
    assert r.error is not None
    assert "thinking" in r.error
    assert r.logprobs is None


@pytest.mark.parametrize(
    ("logprobs", "expected"),
    [
        (None, None),
        ({"content": None}, None),
        (
            {"content": [{"token": "a"}, {"token": "b", "logprob": -0.5, "top_logprobs": None}]},
            [{"token": "b", "logprob": -0.5, "top": {}}],
        ),
    ],
)
def test_partial_logprobs_are_tolerated(key, logprobs, expected) -> None:
    choice = {"message": {"content": "ab"}, "finish_reason": "stop", "logprobs": logprobs}
    with respx.mock(base_url=BASE) as mock:
        mock.post("/chat/completions").respond(json={"model": "m", "choices": [choice]})
        r = mk.text("gpt-4.1-mini", sink=mk.records.NullSink()).complete(
            [{"role": "user", "content": "x"}], logprobs=True
        )
    assert r.text == "ab"
    assert r.logprobs == expected
