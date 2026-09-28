"""Protocols this package owns (design/current.md §8.2). The ports it implements for other packages
(TextClient, DecisionClient, Embedder, GpuLease, Replayer) are satisfied structurally."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

PORTS_VERSION = "1"


class RecordSink(Protocol):
    """Where spans go. Implementations never raise into the caller."""

    def emit(self, span: Mapping[str, Any]) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
