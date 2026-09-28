"""Two processes writing the same store (WAL) must not corrupt it."""

import subprocess
import sys
from pathlib import Path

from hone_models.records import read_spans

WRITER = """
import sys
from hone_models.records import SqliteSpanSink
from hone_models.testing import example_span
sink = SqliteSpanSink(sys.argv[1])
for i in range(50):
    s = example_span()
    s["span_id"] = f"{int(sys.argv[2]):02d}{i:014d}"
    sink.emit(s)
assert sink.failures == 0, sink.failures
"""


def test_two_processes_write_one_store(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    procs = [subprocess.Popen([sys.executable, "-c", WRITER, str(db), str(n)]) for n in (1, 2)]
    assert [p.wait(timeout=60) for p in procs] == [0, 0]
    assert len({s["span_id"] for s in read_spans(db)}) == 100
