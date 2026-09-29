"""Span sinks (design/current.md §8.2): SQLite (default), JSON lines, memory, null.

Every sink applies the same two rules before storing a span:
- content capture: when off (`capture_content=False`, or env `HONE_CAPTURE_CONTENT=0` when the sink
  option is left at `None`), message/output content is replaced by `{"sha256": ..., "len": ...}`;
- secrets: any registered secret value (API keys read from the environment) and anything that looks
  like a bearer token or `sk-` key is replaced by `"***"`.

Sinks never raise into the caller: failures are logged once and counted in `sink.failures`.

    sink = SqliteSpanSink(".hone/models/spans.db")
    mk.text("gemma4-12b", sink=sink)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("hone_models")

CONTENT_KEYS = (
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "hone.models.prompt.variables",
    "hone.models.decision.state",
    "hone.models.decision.questions",
    "hone.models.decision.answers",
    "hone.models.request.params",
    "hone.models.speech.input",
    "hone.models.media.prompt",
    "hone.models.media.revised_prompt",
    "hone.models.media.log_tail",
    "hone.models.media.error",
)
# Attributes whose text values are content (lyrics, texts) while the rest (numbers, file hashes) is not.
TEXT_CONTENT_KEYS = ("hone.models.media.inputs",)
BLOB_THRESHOLD = 64 * 1024
_SECRETS: set[str] = set()
_SECRET_PATTERN = re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+|\bsk-[A-Za-z0-9_-]{16,}")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta   (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS spans (
  span_id        TEXT PRIMARY KEY,
  trace_id       TEXT NOT NULL,
  parent_span_id TEXT,
  name           TEXT NOT NULL,
  kind           TEXT NOT NULL DEFAULT 'internal',
  start_time     TEXT NOT NULL,
  end_time       TEXT,
  status_code    TEXT NOT NULL DEFAULT 'unset',
  status_message TEXT NOT NULL DEFAULT '',
  attributes     TEXT NOT NULL DEFAULT '{}',
  events         TEXT NOT NULL DEFAULT '[]',
  resource       TEXT NOT NULL DEFAULT '{}',
  links          TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS spans_name_time ON spans(name, start_time);
CREATE TABLE IF NOT EXISTS blobs (sha256 TEXT PRIMARY KEY, mime TEXT, size INTEGER, data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS changes (seq INTEGER PRIMARY KEY AUTOINCREMENT, span_id TEXT NOT NULL,
                                    op TEXT NOT NULL, at TEXT NOT NULL);
"""


def add_secret(value: str | None) -> None:
    """Never record `value` (e.g. an API key read from the environment). Short values are ignored."""
    if value and len(value) >= 8:
        _SECRETS.add(value)
        _SECRETS.add(json.dumps(value)[1:-1])  # the form it takes inside serialized JSON


def iso(moment: datetime) -> str:
    """The span time format: UTC ISO-8601 with milliseconds and a `Z` suffix."""
    return moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def now_iso() -> str:
    """The current time in the span time format."""
    return iso(datetime.now(UTC))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(value: Any) -> dict[str, Any]:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return {"sha256": sha256_text(text), "len": len(text)}


def _texts_digested(value: Any) -> Any:
    """A table's text values (also inside lists) replaced by hashes and lengths."""
    if not isinstance(value, Mapping):
        return _digest(value)

    def one(item: Any) -> Any:
        return _digest(item) if isinstance(item, str) else item

    table: Mapping[str, Any] = value  # pyright: ignore[reportUnknownVariableType]
    return {k: [one(i) for i in v] if isinstance(v, list) else one(v) for k, v in table.items()}  # pyright: ignore[reportUnknownVariableType]


def _without_content(span: Mapping[str, Any]) -> dict[str, Any]:
    """Content attributes, error messages and event reasons replaced by hashes and lengths
    (error text can quote prompts or model output)."""
    attrs = {k: _digest(v) if k in CONTENT_KEYS else v for k, v in span.get("attributes", {}).items()}
    attrs.update({k: _texts_digested(attrs[k]) for k in TEXT_CONTENT_KEYS if k in attrs})
    status = dict(span.get("status", {}))
    if status.get("message"):
        status["message"] = json.dumps(_digest(status["message"]))
    events = [
        {
            **e,
            "attributes": {k: _digest(v) if k == "reason" else v for k, v in e.get("attributes", {}).items()},
        }
        for e in span.get("events", [])
    ]
    return {**span, "attributes": attrs, "status": status, "events": events}


def prepare(span: Mapping[str, Any], capture_content: bool) -> dict[str, Any]:
    """A storable copy of `span`: content hashed when capture is off, secrets replaced by `***`."""
    text = json.dumps(span if capture_content else _without_content(span), default=str)
    for secret in _SECRETS:
        text = text.replace(secret, "***")
    text = _SECRET_PATTERN.sub(lambda m: (m.group(1) or "") + "***", text)
    return json.loads(text)


class _Sink:
    """Shared emit logic; subclasses implement `_write`."""

    def __init__(self, capture_content: bool | None = None) -> None:
        self.capture_content = capture_content
        self.failures = 0

    def captures_content(self) -> bool:
        if self.capture_content is not None:
            return self.capture_content
        return os.environ.get("HONE_CAPTURE_CONTENT", "1") != "0"

    def emit(self, span: Mapping[str, Any]) -> None:
        try:
            self._write(prepare(span, self.captures_content()))
        except Exception as exc:  # sinks never break the caller's workflow
            self.failures += 1
            if self.failures == 1:
                log.error("hone-models: could not record span in %s: %s", type(self).__name__, exc)

    def _write(self, span: dict[str, Any]) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        """Writes are immediate; nothing is buffered."""

    def close(self) -> None:
        """Nothing to release by default."""


class NullSink(_Sink):
    """Drops every span."""

    def _write(self, span: dict[str, Any]) -> None:
        pass


class MemorySink(_Sink):
    """Keeps spans in `self.spans` (tests, replay)."""

    def __init__(self, capture_content: bool | None = None) -> None:
        super().__init__(capture_content)
        self.spans: list[dict[str, Any]] = []

    def _write(self, span: dict[str, Any]) -> None:
        self.spans.append(span)


class JsonlSpanSink(_Sink):
    """Appends one JSON object per line to `path`."""

    def __init__(self, path: str | Path, capture_content: bool | None = None) -> None:
        super().__init__(capture_content)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _write(self, span: dict[str, Any]) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(span) + "\n")


def _enable_wal(db: sqlite3.Connection, attempts: int = 50) -> None:
    """Switch to WAL. The switch ignores busy_timeout when another process is creating the same store at
    the same moment, so retry briefly."""
    for attempt in range(attempts):
        try:
            db.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.1)


class SqliteSpanSink(_Sink):
    """Writes spans to the SQLite schema of design/current.md §8.2 (WAL, safe for several processes)."""

    def __init__(self, path: str | Path, capture_content: bool | None = None) -> None:
        super().__init__(capture_content)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self._db.execute("PRAGMA busy_timeout=5000")
        _enable_wal(self._db)
        with self._db:
            self._db.executescript(SCHEMA)
            meta = [("schema", "hone-spans"), ("schema_version", "1"), ("package", "hone-models")]
            meta.append(("created_at", now_iso()))
            self._db.executemany("INSERT OR IGNORE INTO meta VALUES (?, ?)", meta)

    def _write(self, span: dict[str, Any]) -> None:
        with self._lock, self._db:
            attrs = {k: self._maybe_blob(v) for k, v in span.get("attributes", {}).items()}
            status = span.get("status", {})
            self._db.execute(
                "INSERT OR REPLACE INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    span["span_id"],
                    span["trace_id"],
                    span.get("parent_span_id"),
                    span["name"],
                    span.get("kind", "internal"),
                    span["start_time"],
                    span.get("end_time"),
                    status.get("code", "unset"),
                    status.get("message", ""),
                    json.dumps(attrs),
                    json.dumps(span.get("events", [])),
                    json.dumps(span.get("resource", {})),
                    json.dumps(span.get("links", [])),
                ),
            )
            self._db.execute(
                "INSERT INTO changes (span_id, op, at) VALUES (?, 'insert', ?)", (span["span_id"], now_iso())
            )

    def _maybe_blob(self, value: Any) -> Any:
        data = json.dumps(value).encode("utf-8")
        if len(data) <= BLOB_THRESHOLD:
            return value
        digest = hashlib.sha256(data).hexdigest()
        self._db.execute(
            "INSERT OR IGNORE INTO blobs VALUES (?, 'application/json', ?, ?)", (digest, len(data), data)
        )
        return {"$blob": digest}

    def close(self) -> None:
        self._db.close()


def read_spans(path: str | Path) -> list[dict[str, Any]]:
    """Read every span from a SQLite store (oldest first), with blob references resolved."""
    db = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        db.row_factory = sqlite3.Row
        blobs = {r["sha256"]: json.loads(r["data"]) for r in db.execute("SELECT sha256, data FROM blobs")}
        rows = db.execute("SELECT * FROM spans ORDER BY start_time, span_id").fetchall()
    finally:
        db.close()
    spans: list[dict[str, Any]] = []
    for r in rows:
        attrs = json.loads(r["attributes"])
        for key, value in attrs.items():
            if isinstance(value, dict) and set(value) == {"$blob"}:  # pyright: ignore[reportUnknownArgumentType]
                attrs[key] = blobs[value["$blob"]]
        spans.append(
            {
                "trace_id": r["trace_id"],
                "span_id": r["span_id"],
                "parent_span_id": r["parent_span_id"],
                "name": r["name"],
                "kind": r["kind"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "status": {"code": r["status_code"], "message": r["status_message"]},
                "attributes": attrs,
                "events": json.loads(r["events"]),
                "resource": json.loads(r["resource"]),
                "links": json.loads(r["links"]),
            }
        )
    return spans


def default_store() -> Path:
    """`${HONE_HOME:-.hone}/models/spans.db`."""
    return Path(os.environ.get("HONE_HOME", ".hone")) / "models" / "spans.db"


_DEFAULT_SINKS: dict[Path, _Sink] = {}


def default_sink() -> _Sink:
    """The process-wide SQLite sink for `default_store()` (one per path).

    If the store cannot be opened, calls still work: spans are dropped and the error is logged.
    """
    path = default_store().resolve()
    if path not in _DEFAULT_SINKS:
        try:
            _DEFAULT_SINKS[path] = SqliteSpanSink(path)
        except (OSError, sqlite3.Error) as exc:
            log.error("hone-models: cannot open span store %s (%s); calls will not be recorded", path, exc)
            return NullSink()
    return _DEFAULT_SINKS[path]
