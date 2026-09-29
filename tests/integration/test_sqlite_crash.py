"""Crash safety: a writer killed mid-stream leaves a readable, consistent store."""

import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from hone_models.records import SqliteSpanSink, read_spans
from hone_models.testing import example_span

WRITER = """
import sys
from hone_models.records import SqliteSpanSink
from hone_models.testing import example_span
sink = SqliteSpanSink(sys.argv[1])
i = 0
while True:
    s = example_span()
    s["span_id"] = f"{i:016x}"
    s["attributes"]["pad"] = "x" * 5000
    sink.emit(s)
    i += 1
"""


def committed_spans(db: Path) -> int:
    """How many spans the writer has committed so far. The writer creates the file a moment before its
    tables, so a store that is still being created counts as none yet."""
    try:
        return len(read_spans(db))
    except sqlite3.OperationalError:
        return 0


def test_kill_mid_write_then_reopen(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    proc = subprocess.Popen([sys.executable, "-c", WRITER, str(db)])
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if db.exists() and committed_spans(db) > 20:
            break
        time.sleep(0.05)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    sink = SqliteSpanSink(db)
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    spans = read_spans(db)
    assert len(spans) > 20
    assert all(s["attributes"]["pad"] == "x" * 5000 for s in spans)
    assert conn.execute("SELECT count(*) FROM changes").fetchone()[0] == len(spans)
    sink.emit(example_span())
    assert sink.failures == 0


def test_a_store_still_being_created_has_no_committed_spans(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")  # the writer's first step: a file, no tables yet
    conn.close()
    assert committed_spans(db) == 0
