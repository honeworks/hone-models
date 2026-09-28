"""The RecordSink checker itself (AC-17 runs it against every shipped sink)."""

import pytest

from hone_models.records import MemorySink
from hone_models.testing import check_record_sink


def test_checker_rejects_a_sink_that_drops_spans() -> None:
    sink = MemorySink()
    with pytest.raises(AssertionError):
        check_record_sink(sink, list)
