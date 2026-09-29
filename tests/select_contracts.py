"""The contract checkers hone-select owns for the ports hone-models implements.

Imported from hone-select when it is installed (the owners' checkers, design/current.md §10). Otherwise
the same checks run from this copy, so hone-models' own suite doesn't need hone-select installed: the two
packages depend on each other (hone-select's `models` extra), and a dev dependency back on hone-select
would make their lock files pin each other in a cycle.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping as _Mapping
from typing import Any


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, _Mapping) else getattr(obj, key, default)


def check_text_client(client: Any) -> None:
    r = client.complete([{"role": "user", "content": "Say OK."}])
    assert isinstance(_get(r, "text"), str)
    assert _get(r, "error") is None or isinstance(_get(r, "error"), str)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    r = client.complete(
        [{"role": "user", "content": 'Return {"ok": true}.'}],
        schema=schema,
        trace={"traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"},
    )
    assert _get(r, "parsed") is not None or _get(r, "error")
    client.complete([{"role": "user", "content": "x"}], unknown_param_is_ignored=1)


def check_decision_client(client: Any) -> None:
    qs = {
        "q1": {"type": "yes_no", "instructions": "Is the sky described as blue?"},
        "q2": {"type": "choice", "instructions": "Colour?", "options": ["blue", "red"]},
        "q3": {"type": "score", "instructions": "How vivid?", "scale": [1, 5]},
    }
    a = client.decide("The sky is blue.", qs)
    assert set(a) <= set(qs)
    for name, ans in a.items():
        v, err = _get(ans, "value"), _get(ans, "error")
        assert (v is None) == bool(err) or err is None
        if v is not None:
            assert 0.0 <= v <= 1.0
        assert _get(ans, "type") == qs[name]["type"]
        assert isinstance(_get(ans, "calibrated", False), bool)
    if "q2" in a and _get(a["q2"], "value") is not None:
        assert _get(a["q2"], "choice") in ("blue", "red")


def check_embedder(e: Any) -> None:
    v = e.embed(["a", "b"])
    assert len(v) == 2
    assert all(len(x) == e.dimensions for x in v)
    assert all(abs(sum(t * t for t in x) - 1.0) < 1e-3 for x in v)
    assert e.embed([]) == []


def check_machine_probe(probe: Any) -> None:
    """hone-select 0010's `MachineProbe`: `snapshot()` returns a mapping of the documented shape and
    `prepare([])` returns a mapping."""
    snap = probe.snapshot()
    assert isinstance(snap, _Mapping)
    assert {"gpus", "servers", "loaded_models"} <= set(snap)
    assert snap["gpus"] is None or all(
        "memory_total_gb" in g and "utilization_pct" in g for g in snap["gpus"]
    )
    assert all(s["running"] in (True, False, None) for s in snap["servers"])
    assert all({"server", "name", "model_id", "size_gb", "vram_gb"} <= set(m) for m in snap["loaded_models"])
    assert isinstance(probe.prepare([]), _Mapping)


# Prefer the owner's checkers when hone-select is installed.
try:
    _owner: Any = importlib.import_module("hone_select.testing.contracts")
except ImportError:
    _owner = None
if _owner is not None:
    check_text_client = _owner.check_text_client
    check_decision_client = _owner.check_decision_client
    check_embedder = _owner.check_embedder
    check_machine_probe = getattr(_owner, "check_machine_probe", check_machine_probe)

__all__ = ["check_decision_client", "check_embedder", "check_machine_probe", "check_text_client"]
