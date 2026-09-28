"""Model registry: which models exist, how to reach them and what they can do.

TOML files are merged in order: packaged defaults, `~/.config/hone/models.toml`,
`./hone-models.toml`, then explicit paths. Later files override earlier ones key by key.

    reg = load()
    cfg = reg.get("gemma4-12b")                    # registered id
    cfg = reg.get("ollama:llama3.2:1b")            # ad-hoc provider:model id
    cfg = reg.select({"vision": True}, prefer="local")
"""

from __future__ import annotations

import json
import math
import tomllib
from collections.abc import Callable, Iterable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, ValidationError

from .errors import CapabilityError, ConfigError

Provider = Literal["ollama", "openai_compatible", "litellm", "jev", "kokoro", "chatterbox"]
Kind = Literal["chat", "embedding", "decision", "speech"]
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}  # noqa: S104 - a host name to compare, not a bind


class Price(BaseModel):
    """USD per million tokens."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0


class Capabilities(BaseModel):
    """What a model can do. `None` means unknown (not declared and not probed)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    vision: bool | None = None
    thinking: bool | None = None
    json_schema: bool | None = None
    logprobs: bool | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    vram_gb: float | None = None
    license: str | None = None
    dimensions: int | None = None
    questions: list[str] | None = None
    calibrated: bool | None = None
    price: Price | None = None
    speed_tok_s: float | None = None
    voices: list[str] | None = None
    expressive: bool | None = None
    image_tokens: int | None = None  # a flat prompt cost per image (change 0013)
    image_patch_px: int | None = None  # else one token per square of this many pixels (Qwen2.5-VL: 28)


class ModelConfig(BaseModel):
    """One registry entry. `model` is the provider's name for it (defaults to the id)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    provider: Provider
    model: str = ""
    kind: Kind = "chat"
    base_url: str | None = None
    api_key_env: str | None = None
    defaults: dict[str, Any] = {}
    capabilities: Capabilities = Capabilities()
    max_timeout_s: float = 600.0

    @property
    def name(self) -> str:
        """The model name sent to the provider."""
        return self.model or self.id

    @property
    def local(self) -> bool:
        """True when the model runs on this machine (Ollama, or an OpenAI-compatible server on localhost)."""
        if self.provider in ("ollama", "kokoro", "chatterbox"):
            return True
        if self.provider == "openai_compatible" and self.base_url:
            return urlparse(self.base_url).hostname in LOCAL_HOSTS
        return False


# Prefixes accepted in ad-hoc ids ("ollama:llama3.2:1b") and the config they imply.
ADHOC: dict[str, dict[str, Any]] = {
    "ollama": {"provider": "ollama"},
    "openai": {
        "provider": "openai_compatible",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "litellm": {"provider": "litellm"},
    "jev": {"provider": "jev", "kind": "decision", "api_key_env": "TYPESAFE_API_KEY"},
}


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
        elif key in Capabilities.model_fields:
            if getattr(caps, key) != wanted:
                missing.append(f"{key}={wanted}")
        else:
            raise ConfigError(
                f"unknown requirement {key!r}; use min_context or one of {list(Capabilities.model_fields)}"
            )
    return missing


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


class Registry:
    """A merged set of model configs, with lookup and capability-based selection."""

    def __init__(self, models: Mapping[str, ModelConfig]) -> None:
        self.models = dict(models)

    def get(self, model_id: str) -> ModelConfig:
        """Return a registered model (by id or provider model name) or an ad-hoc `provider:model` config."""
        if model_id in self.models:
            return self.models[model_id]
        for cfg in self.models.values():
            if cfg.model == model_id:
                return cfg
        prefix, _, rest = model_id.partition(":")
        if prefix in ADHOC and rest:
            return ModelConfig(id=model_id, model=rest, **ADHOC[prefix])
        raise ConfigError(
            f"unknown model {model_id!r}; registered ids: {sorted(self.models)}; "
            f"or use an ad-hoc id '<provider>:<model>' with provider in {sorted(ADHOC)}"
        )

    def select(
        self, require: Mapping[str, Any] | None = None, prefer: str | None = None, kind: Kind = "chat"
    ) -> ModelConfig:
        """Pick the best registered model of `kind` meeting `require`, ordered by `prefer`.

        `require` keys are capability names (exact value) or `min_context` (max_input_tokens >= n).
        `prefer` is one of "local", "hosted", "cheapest", "fastest" (ties broken by id).
        """
        if prefer is not None and prefer not in PREFER:
            raise ConfigError(f"unknown prefer={prefer!r}; choose one of {sorted(PREFER)}")
        require = dict(require or {})
        candidates = sorted((m for m in self.models.values() if m.kind == kind), key=lambda m: m.id)
        matching = [m for m in candidates if not unmet(m, require)]
        if matching:
            return min(matching, key=PREFER[prefer]) if prefer else matching[0]
        closest = sorted(candidates, key=lambda m: len(unmet(m, require)))[:3]
        listing = "; ".join(f"{m.id} (lacks {', '.join(unmet(m, require))})" for m in closest)
        raise CapabilityError(f"no {kind} model meets {require}; closest candidates: {listing or 'none'}")


def default_paths() -> list[Path]:
    """User and project registry files, in merge order (they may not exist)."""
    return [Path.home() / ".config" / "hone" / "models.toml", Path.cwd() / "hone-models.toml"]


def load(paths: Iterable[str | Path] | str | Path | None = None) -> Registry:
    """Load the packaged defaults, then user and project files if present, then `paths` (must exist)."""
    packaged = resources.files("hone_models").joinpath("data/models.toml").read_text(encoding="utf-8")
    merged: dict[str, Any] = tomllib.loads(packaged).get("models", {})
    if isinstance(paths, str | Path):
        paths = [paths]
    explicit = [Path(p) for p in paths or ()]
    for path in explicit:
        if not path.is_file():
            raise ConfigError(f"registry file {path} does not exist")
    for path in [p for p in default_paths() if p.is_file()] + explicit:
        merged = merge(merged, _read_models(path))
    return Registry({mid: _parse(mid, raw) for mid, raw in merged.items()})


def _read_models(path: Path) -> dict[str, Any]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"registry file {path} is not valid TOML: {exc}") from exc
    unknown = set(data) - {"models"}
    if unknown:
        raise ConfigError(
            f"registry file {path}: unknown top-level keys {sorted(unknown)}; use [models.<id>]"
        )
    return data.get("models", {})


def _parse(model_id: str, raw: Mapping[str, Any]) -> ModelConfig:
    if "id" in raw:
        raise ConfigError(f"registry entry {model_id!r}: remove the 'id' key; the table name is the id")
    try:
        return ModelConfig(id=model_id, **raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid registry entry for {model_id!r}: {exc}") from exc


def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two TOML tables; values in `override` win."""
    out = dict(base)
    for key, value in override.items():
        old = out.get(key)
        if isinstance(old, Mapping) and isinstance(value, Mapping):
            out[key] = merge(old, value)  # pyright: ignore[reportUnknownArgumentType]
        else:
            out[key] = value
    return out


def remember_speed(cfg: ModelConfig, speed_tok_s: float, *, registered: bool) -> Path:
    """Write `capabilities.speed_tok_s` for `cfg` into the user registry and return its path.

    An ad-hoc model gets a full entry. Other entries are kept; comments in the file are not.
    """
    path = default_paths()[0]
    data = _read_models(path) if path.is_file() else {}
    entry = data.get(cfg.id) or ({} if registered else cfg.model_dump(exclude={"id"}, exclude_defaults=True))
    data[cfg.id] = merge(entry, {"capabilities": {"speed_tok_s": round(speed_tok_s, 1)}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_toml(data), encoding="utf-8")
    return path


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
