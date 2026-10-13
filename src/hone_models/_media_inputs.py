"""The media client's checks on one call's named inputs, before any request (change 0015 §1): files that
exist, the sizes, durations and references the entry declares; and how the inputs are recorded."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

from ._media_files import file_record
from .errors import CapabilityError, ConfigError
from .providers.common import FILE_INPUTS
from .registry import ModelConfig

_SIZE = re.compile(r"^(\d+)x(\d+)$")


def as_files(name: str, value: Any) -> Any:
    """File inputs as `Path`s (a string names a file for `references`, `image` and `source`); every file
    must exist."""
    if name in FILE_INPUTS:
        value = [_path(v) for v in items(value)] if isinstance(value, list) else _path(value)
    for item in items(value):
        if isinstance(item, Path) and not item.is_file():
            raise ConfigError(f"input {name!r}: file {item} does not exist")
    return value


def items(value: Any) -> list[Any]:
    """`value` as a list (a single value is a list of one)."""
    return cast(list[Any], value) if isinstance(value, list) else [value]


def _path(value: Any) -> Any:
    return Path(value) if isinstance(value, str) else value


def check_capabilities(cfg: ModelConfig, named: dict[str, Any]) -> None:
    """Refuse what the entry declares impossible (unknown stays allowed)."""
    caps, model = cfg.capabilities, cfg.id
    size = named.get("size")
    if size is not None and not _SIZE.match(str(size)):
        raise ConfigError(f"size must read 'WxH' (e.g. '1024x1024'), got {size!r}")
    if size is not None and caps.sizes is not None and size not in caps.sizes:
        raise CapabilityError(f"model {model!r} makes the sizes {caps.sizes}, not {size!r}")
    duration = named.get("duration_s")
    if duration is not None and caps.durations_s is not None and duration not in caps.durations_s:
        raise CapabilityError(f"model {model!r} makes durations of {caps.durations_s} s, not {duration}")
    if duration is not None and caps.max_duration_s is not None and duration > caps.max_duration_s:
        raise CapabilityError(f"model {model!r} makes at most {caps.max_duration_s} s, not {duration}")
    _check_references(cfg, named.get("references") or [])


def _check_references(cfg: ModelConfig, references: list[Any]) -> None:
    limit = cfg.capabilities.max_references
    if limit is not None and len(references) > limit:
        raise CapabilityError(f"model {cfg.id!r} takes at most {limit} references")
    slots = cfg.inputs.get("references") if isinstance(cfg.inputs, dict) else None
    if cfg.empty_slots == "repeat_last" and slots and not references:
        raise CapabilityError(
            f"model {cfg.id!r} needs at least one reference (its workflow fills every reference slot); "
            "to draw from the prompt alone use a text-to-image model"
        )


def recorded(named: dict[str, Any]) -> dict[str, Any]:
    """The inputs as recorded: files as `{"path", "sha256", "bytes"}`."""

    def one(value: Any) -> Any:
        return file_record(value) if isinstance(value, Path) else value

    return {k: [one(i) for i in items(v)] if isinstance(v, list) else one(v) for k, v in named.items()}
