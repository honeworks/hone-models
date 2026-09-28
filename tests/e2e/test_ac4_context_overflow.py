"""AC-4: prompt + output exceeding the context raises ContextOverflow before any HTTP call."""

import json

import pytest
import respx

import hone_models as mk

pytestmark = pytest.mark.e2e


def test_ac4_context_overflow_before_http() -> None:
    sink = mk.records.MemorySink()
    llm = mk.text("gemma4-12b", sink=sink)  # max_input_tokens = 32768
    long_prompt = "word " * 100_000  # ~142,858 tokens at 3.5 chars/token
    with respx.mock(assert_all_mocked=True) as mock:
        with pytest.raises(mk.errors.ContextOverflow) as info:
            llm.complete([{"role": "user", "content": long_prompt}], max_tokens=1000)
        assert mock.calls.call_count == 0
    message = str(info.value)
    assert "142858" in message  # estimated prompt tokens
    assert "1000" in message  # requested output
    assert "32768" in message  # the limit
    assert "min_context" in message  # what to do
    assert sink.spans[0]["status"]["code"] == "error"
    assert sink.spans[0]["attributes"]["hone.models.context.limit"] == 32768


def test_ac4_exact_fit_is_allowed_and_capped() -> None:
    reply = {"model": "m", "message": {"content": "ok"}, "done_reason": "stop"}
    with respx.mock(base_url="http://127.0.0.1:11434") as mock:
        chat = mock.post("/api/chat").respond(json=reply)
        mk.text("gemma4-12b", sink=mk.records.NullSink()).complete(
            [{"role": "user", "content": "short"}], max_tokens=32766
        )
    assert json.loads(chat.calls.last.request.content)["options"]["num_ctx"] == 32768


def test_ac4_schema_instruction_counts_toward_the_budget() -> None:
    big_schema = {"type": "object", "properties": {f"field_{i}": {"type": "string"} for i in range(3000)}}
    llm = mk.text("gemma4-12b", sink=mk.records.NullSink())
    with respx.mock(assert_all_mocked=True), pytest.raises(mk.errors.ContextOverflow):
        llm.complete([{"role": "user", "content": "short"}], schema=big_schema, max_tokens=20000)


def test_ac4_invalid_max_tokens() -> None:
    llm = mk.text("gemma4-12b", sink=mk.records.NullSink())
    with respx.mock(assert_all_mocked=True), pytest.raises(mk.errors.ConfigError, match="max_tokens must be"):
        llm.complete([{"role": "user", "content": "x"}], max_tokens=0)


def test_ac4_output_budget_counts_too() -> None:
    llm = mk.text("gemma4-12b", sink=mk.records.NullSink())
    with (
        respx.mock(assert_all_mocked=True),
        pytest.raises(mk.errors.ContextOverflow, match="32769 tokens exceeds"),
    ):
        llm.complete([{"role": "user", "content": "short"}], max_tokens=32767)
