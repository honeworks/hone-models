import base64
import json
from pathlib import Path

import pytest
import respx

import hone_models as mk
from hone_models.errors import CapabilityError, ConfigError
from hone_models.registry import load
from hone_models.text import cost_usd

URL = "http://127.0.0.1:11434"
OK = {
    "model": "m",
    "message": {"content": "fine"},
    "done_reason": "stop",
    "prompt_eval_count": 3,
    "eval_count": 2,
}


def test_image_path_is_sent_as_base64(isolated: Path, recorded) -> None:
    png = isolated / "frame.png"
    png.write_bytes(b"\x89PNG fake")
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/show").respond(json=recorded("ollama_show_vision.json"))
        chat = mock.post("/api/chat").respond(json=OK)
        llm = mk.text("ollama:qwen2.5vl:3b", sink=mk.records.NullSink())
        llm.complete(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Describe."}, {"type": "image", "path": str(png)}],
                }
            ]
        )
    sent = json.loads(chat.calls.last.request.content)["messages"][0]
    assert sent == {
        "role": "user",
        "content": "Describe.",
        "images": [base64.b64encode(b"\x89PNG fake").decode()],
    }
    assert llm.config.capabilities.vision is True
    with respx.mock(base_url=URL, assert_all_called=False) as mock:
        show = mock.post("/api/show")
        mock.post("/api/chat").respond(json=OK)
        llm.complete([{"role": "user", "content": "again"}])
    assert show.call_count == 0  # probed once, cached


def test_inline_image_data_and_missing_image(isolated: Path) -> None:
    llm = mk.text("qwen2.5vl-7b", sink=mk.records.NullSink())
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat").respond(json=OK)
        llm.complete(
            [{"role": "user", "content": [{"type": "image", "data_b64": "QUJD", "mime": "image/png"}]}]
        )
    assert json.loads(chat.calls.last.request.content)["messages"][0]["images"] == ["QUJD"]
    with pytest.raises(ConfigError, match="does not exist"):
        llm.complete([{"role": "user", "content": [{"type": "image", "path": str(isolated / "nope.png")}]}])


def test_defaults_merge_and_unknown_params_are_ignored() -> None:
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat").respond(json=OK)
        mk.text("gemma4-12b", sink=mk.records.NullSink()).complete(
            [{"role": "user", "content": "x"}], top_p=0.5, max_tokens=64, unknown_param_is_ignored=1
        )
    options = json.loads(chat.calls.last.request.content)["options"]
    assert options["temperature"] == 0.8  # registry default
    assert options["top_p"] == 0.5
    assert options["num_predict"] == 64
    assert "unknown_param_is_ignored" not in options


def test_factory_errors() -> None:
    with pytest.raises(ConfigError, match="is a embedding model"):
        mk.text("nomic-embed-text")
    with pytest.raises(CapabilityError, match="does not meet vision=True"):
        mk.text("gemma4-12b", require={"vision": True})
    assert mk.text(require={"vision": True}, prefer="local").model_id == "qwen2.5vl-7b"


def test_cost_from_price() -> None:
    cfg = load().get("gpt-4.1-mini")
    assert cost_usd(cfg, {"input_tokens": 1_000_000, "output_tokens": 500_000}) == pytest.approx(1.2)
    assert cost_usd(load().get("gemma4-12b"), {"input_tokens": 5}) is None
    assert cost_usd(cfg, {"input_tokens": 5}) is None  # output is priced but its count is unknown


def test_cost_needs_only_the_priced_token_counts() -> None:
    """Jev prices input only; a reply that reports no output tokens still has a cost."""
    jev = load().get("jev")
    assert cost_usd(jev, {"input_tokens": 1_000_000}) == pytest.approx(0.042)
    assert cost_usd(jev, {}) is None
    assert cost_usd(jev, {"input_tokens": 0}) == 0.0  # known and free, not unknown
    assert cost_usd(jev, {"output_tokens": 10}) is None  # the priced input count is missing
    assert cost_usd(load().get("gpt-4.1-mini"), {}) is None


def test_trace_context_links_call_to_caller() -> None:
    sink = mk.records.MemorySink()
    tp = "00-" + "c" * 32 + "-" + "d" * 16 + "-01"
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(json=OK)
        mk.text("gemma4-12b", sink=sink).complete(
            [{"role": "user", "content": "x"}], trace={"traceparent": tp, "hone.run_id": "run-1"}
        )
    span = sink.spans[0]
    assert (span["trace_id"], span["parent_span_id"]) == ("c" * 32, "d" * 16)
    assert span["attributes"]["hone.run_id"] == "run-1"


def test_probe_errors_and_unknown_context() -> None:
    with respx.mock(base_url=URL, assert_all_called=False) as mock:
        mock.post("/api/show").respond(404, json={"error": "model 'nope' not found"})
        chat = mock.post("/api/chat")
        with pytest.raises(mk.errors.ProviderError) as info:
            mk.text("ollama:nope", sink=mk.records.NullSink()).complete([{"role": "user", "content": "x"}])
        assert info.value.status == 404
        assert chat.call_count == 0
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/show").respond(json={"capabilities": ["completion"]})
        chat = mock.post("/api/chat").respond(json=OK)
        llm = mk.text("ollama:tiny", sink=mk.records.NullSink())
        llm.complete(
            [{"role": "user", "content": "x" * 350_000}], max_tokens=100
        )  # no limit known: no overflow; num_ctx = 100100 * 1.1 rounded up to 2048s
    assert llm.config.capabilities.max_input_tokens is None
    assert json.loads(chat.calls.last.request.content)["options"]["num_ctx"] == 110592


def test_image_part_without_data_is_a_config_error() -> None:
    llm = mk.text("qwen2.5vl-7b", sink=mk.records.NullSink())
    with respx.mock(assert_all_mocked=True), pytest.raises(ConfigError, match='needs "path" or "data_b64"'):
        llm.complete([{"role": "user", "content": [{"type": "image"}]}])


def test_explicit_think_false_is_sent_and_ollama_host_is_used(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.5:11434")
    with respx.mock(base_url="http://10.0.0.5:11434") as mock:
        chat = mock.post("/api/chat").respond(json=OK)
        mk.text("qwen2.5vl-7b", sink=mk.records.NullSink()).complete(
            [{"role": "user", "content": "x"}], think=False
        )
    assert json.loads(chat.calls.last.request.content)["think"] is False


def test_provider_without_chat_is_a_config_error(isolated: Path) -> None:
    path = isolated / "r.toml"
    path.write_text('[models.odd]\nprovider = "jev"\nkind = "chat"\n')
    llm = mk.text("odd", registry=mk.registry.load([path]), sink=mk.records.NullSink())
    with pytest.raises(ConfigError, match=r"provider 'jev' of model 'odd' does not support chat"):
        llm.complete([{"role": "user", "content": "x"}])
