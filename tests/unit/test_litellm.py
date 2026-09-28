import json
import sys
from types import SimpleNamespace

import pytest

import hone_models as mk
from hone_models.errors import ConfigError, ModelTimeout, ProviderError

litellm = pytest.importorskip("litellm")

REPLY = {
    "model": "anthropic/claude-sonnet-5",
    "choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 4},
}


class FakeLiteLLM:
    """Stands in for `litellm.completion` / `litellm.embedding`; records the keyword arguments."""

    def __init__(self, reply=None, error: Exception | None = None) -> None:
        self.reply, self.error, self.calls = reply, error, []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(model_dump=lambda: self.reply)


def test_chat_through_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLiteLLM(REPLY)
    monkeypatch.setattr(litellm, "completion", fake)
    monkeypatch.setattr(litellm, "get_model_info", lambda name: {"supports_response_schema": True})
    sink = mk.records.MemorySink()
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    r = mk.text("litellm:anthropic/claude-sonnet-5", sink=sink).complete(
        [{"role": "user", "content": "ok?"}], schema=schema, temperature=0.2
    )
    assert (r.parsed, r.structured_path, r.usage) == (
        {"ok": True},
        "constrained",
        {"input_tokens": 12, "output_tokens": 4},
    )
    sent = fake.calls[0]
    assert sent["model"] == "anthropic/claude-sonnet-5"
    assert sent["temperature"] == 0.2
    assert sent["response_format"]["json_schema"]["schema"] == schema
    assert sent["num_retries"] == 2
    assert "api_key" not in sent
    assert sink.spans[0]["attributes"]["gen_ai.provider.name"] == "litellm"


def test_capability_hints_from_the_model_map(monkeypatch: pytest.MonkeyPatch) -> None:
    info = {
        "supports_vision": False,
        "max_input_tokens": 100,
        "max_output_tokens": 50,
        "input_cost_per_token": 3e-6,
        "output_cost_per_token": 1.5e-5,
    }
    monkeypatch.setattr(litellm, "get_model_info", lambda name: info)
    monkeypatch.setattr(litellm, "completion", FakeLiteLLM(REPLY))
    sink = mk.records.MemorySink()
    llm = mk.text("litellm:some/model", sink=sink)
    with pytest.raises(mk.errors.CapabilityError, match="cannot read images"):
        llm.complete(
            [{"role": "user", "content": [{"type": "image", "data_b64": "QQ==", "mime": "image/png"}]}]
        )
    with pytest.raises(mk.errors.ContextOverflow):
        llm.complete([{"role": "user", "content": "x" * 1000}])
    r = llm.complete([{"role": "user", "content": "hi"}])
    assert sink.spans[-1]["attributes"]["hone.models.cost_usd"] == pytest.approx((12 * 3 + 4 * 15) / 1e6)
    assert r.text == '{"ok": true}'


def test_unmapped_model_has_unknown_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    def unmapped(name):
        raise Exception("This model isn't mapped yet.")

    monkeypatch.setattr(litellm, "get_model_info", unmapped)
    monkeypatch.setattr(litellm, "completion", FakeLiteLLM(REPLY))
    llm = mk.text("litellm:new/model", sink=mk.records.NullSink())
    assert llm.complete([{"role": "user", "content": "hi"}]).error is None
    assert llm.config.capabilities == mk.registry.Capabilities()


@pytest.mark.parametrize(
    ("error", "raised", "status"),
    [
        (litellm.exceptions.Timeout("slow", model="m", llm_provider="x"), ModelTimeout, None),
        (litellm.exceptions.BadRequestError("bad", model="m", llm_provider="x"), ProviderError, 400),
    ],
)
def test_litellm_errors_are_mapped(monkeypatch, error, raised, status) -> None:
    monkeypatch.setattr(litellm, "completion", FakeLiteLLM(error=error))
    monkeypatch.setattr(litellm, "get_model_info", lambda name: {})
    with pytest.raises(raised) as info:
        mk.text("litellm:m", sink=mk.records.NullSink()).complete([{"role": "user", "content": "hi"}])
    assert getattr(info.value, "status", None) == status


def test_registered_model_with_api_key_and_embeddings(isolated, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "key-000000000000")
    path = isolated / "reg.toml"
    path.write_text(
        '[models.emb]\nprovider = "litellm"\nmodel = "voyage/voyage-3"\nkind = "embedding"\n'
        'api_key_env = "MY_KEY"\nbase_url = "https://proxy.example/v1"\n'
    )
    fake = FakeLiteLLM({"data": [{"index": 0, "embedding": [3.0, 4.0]}]})
    monkeypatch.setattr(litellm, "embedding", fake)
    sink = mk.records.MemorySink()
    vecs = mk.embedder("emb", registry=mk.registry.load([path]), sink=sink).embed(["a"])
    assert vecs == [[0.6, 0.8]]
    assert (fake.calls[0]["api_key"], fake.calls[0]["api_base"]) == (
        "key-000000000000",
        "https://proxy.example/v1",
    )
    assert "key-000000000000" not in json.dumps(sink.spans)


def test_missing_extra_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "litellm", None)  # makes `import litellm` fail
    with pytest.raises(ConfigError, match=r"hone-models\[litellm\]"):
        mk.text("litellm:m", sink=mk.records.NullSink()).complete([{"role": "user", "content": "hi"}])
