"""Writing registry tables back as TOML (for `remember_speed`)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def to_toml(models: Mapping[str, Mapping[str, Any]]) -> str:
    """`[models."<id>"]` tables; nested tables are written inline, `None` values are left out."""
    lines: list[str] = []
    for model_id, entry in models.items():
        lines += [f"[models.{json.dumps(model_id)}]", *_pairs(entry), ""]
    return "\n".join(lines)


def _pairs(table: Mapping[str, Any]) -> list[str]:
    return [f"{json.dumps(k)} = {_toml_value(v)}" for k, v in table.items() if v is not None]


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        return "{" + ", ".join(_pairs(value)) + "}"  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"  # pyright: ignore[reportUnknownVariableType]
    # str, int, float: JSON and TOML spell them the same, except that TOML allows no surrogate-pair
    # escapes (so non-ASCII stays literal) and wants DEL escaped.
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
