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


def test_kill_mid_write_then_reopen(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    proc = subprocess.Popen([sys.executable, "-c", WRITER, str(db)])
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if db.exists() and len(read_spans(db)) > 20:
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
