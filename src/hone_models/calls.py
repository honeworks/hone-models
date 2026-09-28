"""Recorded model calls: the filters and statistics behind `hone-models calls ...`.

spans = mk.records.read_spans(".hone/models/spans.db")
recent = find_calls(spans, since="1d", model="gemma4-12b")
call_stats(recent, by="model")   # [{"key": "gemma4-12b", "calls": 12, "errors": 1, ...}]
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from .errors import ConfigError
from .records import iso

CALL_SPANS = ("hone.models.chat", "hone.models.decide", "hone.models.embed")
GROUP_BY = {"model": "hone.models.model_id", "provider": "gen_ai.provider.name", "tag": "hone.step"}
UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
_DURATION = re.compile(r"^(\d+)([mhdw])$")


def since_cutoff(since: str, now: datetime | None = None) -> str:
    """`"30m"`, `"12h"`, `"1d"`, `"2w"` or an ISO-8601 time -> the earliest start time to keep."""
    match = _DURATION.match(since)
    if match:
        delta = timedelta(**{UNITS[match.group(2)]: int(match.group(1))})
        return iso((now or datetime.now(UTC)) - delta)
    try:
        moment = datetime.fromisoformat(since)
    except ValueError:
        raise ConfigError(f"--since {since!r}: use a duration like 30m, 12h, 1d, 2w or an ISO time") from None
    return iso(moment if moment.tzinfo else moment.replace(tzinfo=UTC))


def find_calls(
    spans: Iterable[Mapping[str, Any]], *, since: str | None = None, model: str | None = None
) -> list[Mapping[str, Any]]:
    """Model-call spans, optionally since `since` (see `since_cutoff`) and for one model (registry id or
    provider model name).

    A call that made another call (an emulated decision and its chat) is left out, so each model request
    is counted once, with its tokens and cost.
    """
    calls = [s for s in spans if s["name"] in CALL_SPANS]
    parents = {s["parent_span_id"] for s in calls}
    cutoff = since_cutoff(since) if since else None
    out: list[Mapping[str, Any]] = []
    for span in calls:
        attrs = span["attributes"]
        names = (attrs.get("hone.models.model_id"), attrs.get("gen_ai.request.model"))
        if span["span_id"] in parents or (cutoff and span["start_time"] < cutoff):
            continue
        if model is None or model in names:
            out.append(span)
    return out


def duration_ms(span: Mapping[str, Any]) -> float | None:
    """End minus start in milliseconds (`None` for an unfinished span)."""
    if not span.get("end_time"):
        return None
    start, end = (datetime.fromisoformat(span[k]) for k in ("start_time", "end_time"))
    return (end - start).total_seconds() * 1000


def call_stats(spans: Iterable[Mapping[str, Any]], by: str) -> list[dict[str, Any]]:
    """Per group: calls, errors, input/output tokens, cost (`None` when no call reported one), mean ms."""
    if by not in GROUP_BY:
        raise ConfigError(f"--by {by!r}: use one of {sorted(GROUP_BY)}")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for span in spans:
        groups.setdefault(str(span["attributes"].get(GROUP_BY[by], "-")), []).append(span)
    return [_stats(key, group) for key, group in sorted(groups.items())]


def _stats(key: str, spans: list[Mapping[str, Any]]) -> dict[str, Any]:
    attrs = [s["attributes"] for s in spans]
    costs = [a["hone.models.cost_usd"] for a in attrs if "hone.models.cost_usd" in a]
    times = [ms for ms in map(duration_ms, spans) if ms is not None]
    return {
        "key": key,
        "calls": len(spans),
        "errors": sum(s["status"]["code"] == "error" for s in spans),
        "input_tokens": sum(a.get("gen_ai.usage.input_tokens", 0) for a in attrs),
        "output_tokens": sum(a.get("gen_ai.usage.output_tokens", 0) for a in attrs),
        "cost_usd": sum(costs) if costs else None,
        "mean_ms": round(sum(times) / len(times), 1) if times else None,
    }
