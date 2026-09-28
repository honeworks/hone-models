"""Contract checker for the port this package owns (`RecordSink`, design/current.md §8.2)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any


def example_span() -> dict[str, Any]:
    """An example span in the shape of design/current.md §8.1."""
    return {
        "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
        "span_id": "00f067aa0ba902b7",
        "parent_span_id": None,
        "name": "hone.test.example",
        "kind": "internal",
        "start_time": "2026-09-27T14:03:11.120Z",
        "end_time": "2026-09-27T14:03:11.220Z",
        "status": {"code": "ok", "message": ""},
        "attributes": {"hone.schema_version": "1"},
        "events": [],
        "resource": {},
        "links": [],
    }


def check_record_sink(sink: Any, read_back: Callable[[], Iterable[Mapping[str, Any]]]) -> None:
    """Emit the example span and assert it can be read back with the same trace id."""
    span = example_span()
    sink.emit(span)
    sink.flush()
    got = [s for s in read_back() if s["span_id"] == span["span_id"]]
    assert got and got[0]["trace_id"] == span["trace_id"]
