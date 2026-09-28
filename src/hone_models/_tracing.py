"""Trace context (design/current.md §8.5) and span creation (§8.1).

with start_span("hone.models.chat", sink, {"gen_ai.operation.name": "chat"}, trace=trace) as span:
    ...  # nested spans inherit the trace id and use this span as parent
"""

from __future__ import annotations

import os
import re
import secrets
import socket
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from . import __version__
from .ports import RecordSink
from .records import log, now_iso

# design/current.md §8.3: copied from the trace context onto every span (a replay carries its finding id)
SHARED_KEYS = (
    "hone.run_id",
    "hone.item",
    "hone.step",
    "hone.candidate_id",
    "hone.scorer",
    "hone.lens.finding_id",
)
_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")

_trace: ContextVar[dict[str, str] | None] = ContextVar("hone_models_trace", default=None)
_span: ContextVar[dict[str, Any] | None] = ContextVar("hone_models_span", default=None)
# Extra attributes set by an enclosing scope (the GPU lease) for every span made inside it.
scope_attributes: ContextVar[dict[str, Any] | None] = ContextVar("hone_models_scope_attrs", default=None)


@contextmanager
def extra_attributes(attrs: Mapping[str, Any]) -> Generator[None]:
    """Add `attrs` to every span started inside the block."""
    token = scope_attributes.set({**(scope_attributes.get() or {}), **attrs})
    try:
        yield
    finally:
        scope_attributes.reset(token)


def current_trace() -> dict[str, str]:
    """The active trace context (empty when none), to pass to other packages as `trace=`."""
    return dict(_trace.get() or {})


def _resource() -> dict[str, Any]:
    return {
        "service.name": os.environ.get("OTEL_SERVICE_NAME", "hone-models"),
        "hone.package": "hone-models",
        "hone.package.version": __version__,
        "host.name": socket.gethostname(),
        "process.pid": os.getpid(),
    }


@contextmanager
def start_span(
    name: str,
    sink: RecordSink,
    attributes: Mapping[str, Any] | None = None,
    *,
    trace: Mapping[str, str] | None = None,
    kind: str = "client",
) -> Generator[dict[str, Any]]:
    """Open a span; on exit set status (error on exception) and emit it to `sink`."""
    ctx = dict(trace) if trace is not None else current_trace()
    match = _TRACEPARENT.match(ctx.get("traceparent", "").lower())
    trace_id, parent_id = (match.group(1), match.group(2)) if match else (secrets.token_hex(16), None)
    span_id = secrets.token_hex(8)
    attrs: dict[str, Any] = {"hone.schema_version": "1"}
    attrs.update({k: ctx[k] for k in SHARED_KEYS if k in ctx})
    attrs.update(scope_attributes.get() or {})
    attrs.update(attributes or {})
    span: dict[str, Any] = {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_id,
        "name": name,
        "kind": kind,
        "start_time": now_iso(),
        "end_time": None,
        "status": {"code": "unset", "message": ""},
        "attributes": attrs,
        "events": [],
        "resource": _resource(),
        "links": [],
    }
    trace_token = _trace.set({**ctx, "traceparent": f"00-{trace_id}-{span_id}-01"})
    span_token = _span.set(span)
    try:
        yield span
    except BaseException as exc:
        span["status"] = {"code": "error", "message": str(exc) or type(exc).__name__}
        raise
    finally:
        _span.reset(span_token)
        _trace.reset(trace_token)
        span["end_time"] = now_iso()
        if span["status"]["code"] == "unset":
            span["status"]["code"] = "ok"
        try:
            sink.emit(span)
        except Exception:  # a third-party sink broke its "never raise" contract; don't hide the call
            log.exception("hone-models: record sink %r raised", sink)


def add_event(name: str, attributes: Mapping[str, Any]) -> None:
    """Attach an event (e.g. a retry) to the innermost open span, if any."""
    span = _span.get()
    if span is not None:
        span["events"].append({"name": name, "time": now_iso(), "attributes": dict(attributes)})
