import pytest

from hone_models import current_trace
from hone_models._tracing import add_event, scope_attributes, start_span
from hone_models.records import MemorySink

TP = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"


def test_new_trace_and_nested_parent() -> None:
    sink = MemorySink()
    with start_span("outer", sink) as outer, start_span("inner", sink, {"k": 1}) as inner:
        assert current_trace()["traceparent"] == f"00-{outer['trace_id']}-{inner['span_id']}-01"
    assert current_trace() == {}
    inner_rec, outer_rec = sink.spans
    assert len(outer_rec["trace_id"]) == 32
    assert len(outer_rec["span_id"]) == 16
    assert outer_rec["parent_span_id"] is None
    assert inner_rec["trace_id"] == outer_rec["trace_id"]
    assert inner_rec["parent_span_id"] == outer_rec["span_id"]
    assert inner_rec["attributes"] == {"hone.schema_version": "1", "k": 1}
    assert inner_rec["status"] == {"code": "ok", "message": ""}
    assert inner_rec["kind"] == "client"
    assert inner_rec["resource"]["hone.package"] == "hone-models"
    assert inner_rec["start_time"].endswith("Z")


def test_incoming_trace_context_and_shared_keys() -> None:
    sink = MemorySink()
    with start_span("s", sink, trace={"traceparent": TP, "hone.run_id": "r1", "other": "x"}):
        pass
    [s] = sink.spans
    assert (s["trace_id"], s["parent_span_id"]) == ("a" * 32, "b" * 16)
    assert s["attributes"]["hone.run_id"] == "r1"
    assert "other" not in s["attributes"]


def test_error_status_and_events() -> None:
    sink = MemorySink()

    def failing_call() -> None:
        with start_span("s", sink):
            add_event("retry", {"attempt": 2})
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        failing_call()
    [s] = sink.spans
    assert s["status"] == {"code": "error", "message": "boom"}
    assert s["events"][0]["name"] == "retry"
    assert s["events"][0]["attributes"] == {"attempt": 2}
    add_event("ignored", {})  # no open span: no error


def test_scope_attributes_apply_to_spans_inside() -> None:
    sink = MemorySink()
    token = scope_attributes.set({"hone.models.gpu.lease_wait_ms": 5})
    with start_span("s", sink):
        pass
    scope_attributes.reset(token)
    assert sink.spans[0]["attributes"]["hone.models.gpu.lease_wait_ms"] == 5


class RaisingSink:
    def emit(self, span: object) -> None:
        raise RuntimeError("sink broke")

    def flush(self) -> None: ...
    def close(self) -> None: ...


def test_a_raising_sink_does_not_break_or_mask_the_call(caplog: pytest.LogCaptureFixture) -> None:
    with start_span("s", RaisingSink()):
        pass
    assert "sink" in caplog.text

    def body_fails() -> None:
        with start_span("s", RaisingSink()):
            raise ValueError("original")

    with pytest.raises(ValueError, match="original"):
        body_fails()


def test_malformed_traceparent_starts_new_root_and_uppercase_is_accepted() -> None:
    sink = MemorySink()
    with start_span("s", sink, trace={"traceparent": "garbage"}):
        pass
    with start_span("s", sink, trace={"traceparent": TP.upper().replace("00-", "00-", 1)}):
        pass
    assert sink.spans[0]["parent_span_id"] is None
    assert sink.spans[1]["trace_id"] == "a" * 32
