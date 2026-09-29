"""Selecting registry models by capability: which requirements an entry misses, and the orderings."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ._registry_shapes import Capabilities
from .errors import ConfigError

if TYPE_CHECKING:
    from .registry import ModelConfig


def unmet(cfg: ModelConfig, require: Mapping[str, Any]) -> list[str]:
    """The requirements `cfg` does not meet, as short readable strings."""
    caps = cfg.capabilities
    missing: list[str] = []
    for key, wanted in require.items():
        if key == "min_context":
            if not isinstance(wanted, int) or isinstance(wanted, bool):
                raise ConfigError(f"min_context must be an int, got {wanted!r}")
            if (caps.max_input_tokens or 0) < wanted:
                missing.append(f"min_context={wanted}")
        elif key == "features":
            missing += _missing_features(caps.features or [], wanted)
        elif key in Capabilities.model_fields:
            if getattr(caps, key) != wanted:
                missing.append(f"{key}={wanted}")
        else:
            raise ConfigError(
                f"unknown requirement {key!r}; use min_context or one of {list(Capabilities.model_fields)}"
            )
    return missing


def _missing_features(have: list[str], wanted: Any) -> list[str]:
    """`features=[...]` is met when the model declares every listed feature (case ignored)."""
    names = [wanted] if isinstance(wanted, str) else list(wanted)
    declared = {name.casefold() for name in have}
    return [f"features ∋ {name!r}" for name in names if str(name).casefold() not in declared]


def _price(cfg: ModelConfig) -> float:
    if cfg.local:
        return 0.0
    price = cfg.capabilities.price
    return price.input_per_mtok + price.output_per_mtok if price else math.inf


PREFER: dict[str, Callable[[ModelConfig], Any]] = {
    "local": lambda m: (not m.local, m.id),
    "hosted": lambda m: (m.local, m.id),
    "cheapest": lambda m: (_price(m), m.id),
    "fastest": lambda m: (m.capabilities.speed_tok_s is None, -(m.capabilities.speed_tok_s or 0.0), m.id),
}
