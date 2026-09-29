"""Model registry: which models exist, how to reach them and what they can do.

TOML files are merged in order: packaged defaults, `~/.config/hone/models.toml`,
`./hone-models.toml`, then explicit paths. Later files override earlier ones key by key.

    reg = load()
    cfg = reg.get("gemma4-12b")                    # registered id
    cfg = reg.get("ollama:llama3.2:1b")            # ad-hoc provider:model id
    cfg = reg.select({"vision": True}, prefer="local")
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Iterable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from ._registry_select import PREFER
from ._registry_select import unmet as unmet  # noqa: PLC0414 - re-exported for text.py
from ._registry_shapes import (
    Capabilities,
    Guide,
    Install,
    NodeMapping,
    PromptInput,
    check_node_mapping,
    feature_names,
    shape_errors,
)
from ._registry_shapes import Price as Price  # noqa: PLC0414 - re-exported: `hone_models.registry.Price`
from ._toml_write import to_toml as to_toml  # noqa: PLC0414 - re-exported: `hone_models.registry.to_toml`
from .errors import CapabilityError, ConfigError

Provider = Literal[
    "ollama", "openai_compatible", "litellm", "jev", "kokoro", "chatterbox",
    "comfyui", "command", "faster_whisper", "none",
]  # fmt: skip  # "none": a catalog entry hone-models cannot call yet (§3b)
Kind = Literal[
    "chat", "embedding", "decision", "speech", "image", "music", "video", "transcription", "scoring",
]  # fmt: skip
# Kinds in the catalog that hone-models cannot call yet (change 0015 §3b): calling one is a `ConfigError`.
NO_CLIENT_KINDS = ("scoring",)
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}  # noqa: S104 - a host name to compare, not a bind
MEDIA_KINDS = ("image", "music", "video")
# `max_timeout_s` when an entry does not set it (change 0015 §3): generation takes minutes.
KIND_TIMEOUT_S = {"image": 600.0, "music": 1800.0, "video": 3600.0, "transcription": 600.0}
COMFYUI_URL = "http://127.0.0.1:8188"
# Entry keys of generation models, all `None` on other entries.
GENERATION_KEYS = (
    "workflow", "inputs", "outputs", "command", "cwd", "env",
    "lyrics_format", "prompt_inputs", "guide", "install",
)  # fmt: skip


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
    # generation entries (change 0015 §3): `inputs` is a table of workflow paths for `comfyui`, a list of
    # extra input names for the other providers
    workflow: str | None = None  # an API-format ComfyUI workflow; absolute once loaded
    inputs: dict[str, NodeMapping] | list[str] | None = None
    outputs: list[str] | None = None
    command: list[str] | None = None
    cwd: str | None = None
    env: dict[str, str] | None = None
    lyrics_format: Literal["sections", "levo", "heartmula", "plain"] | None = None
    prompt_inputs: dict[str, PromptInput] | None = None
    guide: Guide | None = None
    install: Install | None = None

    @model_validator(mode="before")
    @classmethod
    def _by_kind_and_provider(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw: dict[str, Any] = dict(data)  # pyright: ignore[reportUnknownArgumentType]
        errors = shape_errors(str(raw.get("provider")), raw)
        if errors:
            raise ValueError("; ".join(errors))
        raw.setdefault("max_timeout_s", KIND_TIMEOUT_S.get(str(raw.get("kind", "chat")), 600.0))
        features = feature_names(raw.get("guide"))
        if features:  # selection matches the guide's features (§3a); an explicit list wins
            given: Any = raw.get("capabilities") or {}
            caps: dict[str, Any] = (
                given.model_dump(exclude_none=True) if isinstance(given, Capabilities) else dict(given)
            )
            raw["capabilities"] = {"features": features, **caps}
        return raw

    @field_validator("inputs")
    @classmethod
    def _inputs(cls, value: dict[str, NodeMapping] | list[str] | None) -> Any:
        return check_node_mapping(value) if isinstance(value, dict) else value

    @property
    def name(self) -> str:
        """The model name sent to the provider."""
        return self.model or self.id

    @property
    def local(self) -> bool:
        """True when the model runs on this machine: Ollama, the in-process and `command` models, and a
        ComfyUI or OpenAI-compatible server on this host."""
        if self.provider in ("ollama", "kokoro", "chatterbox", "command", "faster_whisper", "none"):
            return True
        if self.provider == "comfyui":
            return urlparse(self.comfyui_url).hostname in LOCAL_HOSTS
        if self.provider == "openai_compatible" and self.base_url:
            return urlparse(self.base_url).hostname in LOCAL_HOSTS
        return False

    @property
    def comfyui_url(self) -> str:
        """A `comfyui` entry's server: `base_url`, else `HONE_COMFYUI_URL`, else `COMFYUI_URL`."""
        return (self.base_url or os.environ.get("HONE_COMFYUI_URL") or COMFYUI_URL).rstrip("/")


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


def require_client(cfg: ModelConfig) -> None:
    """A `ConfigError` for a catalog entry of a kind hone-models cannot call yet (§3b)."""
    if cfg.kind in NO_CLIENT_KINDS:
        raise ConfigError(
            f"no client for kind {cfg.kind!r} yet: model {cfg.id!r} is in the catalog (see "
            f"`hone-models models guide {cfg.id}`), but hone-models cannot call it yet"
        )


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

        `require` keys are capability names (exact value), `min_context` (max_input_tokens >= n) or
        `features` (every listed feature declared).
        `prefer` is one of "local", "hosted", "cheapest", "fastest" (ties broken by id).
        """
        matching = self.matching(require, prefer, kind)
        if matching:
            return matching[0]
        require = dict(require or {})
        candidates = sorted((m for m in self.models.values() if m.kind == kind), key=lambda m: m.id)
        closest = sorted(candidates, key=lambda m: len(unmet(m, require)))[:3]
        listing = "; ".join(f"{m.id} (lacks {', '.join(unmet(m, require))})" for m in closest)
        raise CapabilityError(f"no {kind} model meets {require}; closest candidates: {listing or 'none'}")

    def matching(
        self, require: Mapping[str, Any] | None = None, prefer: str | None = None, kind: Kind = "chat"
    ) -> list[ModelConfig]:
        """Every registered model of `kind` meeting `require`, best first (by `prefer`, then id)."""
        if prefer is not None and prefer not in PREFER:
            raise ConfigError(f"unknown prefer={prefer!r}; choose one of {sorted(PREFER)}")
        require = dict(require or {})
        found = sorted((m for m in self.models.values() if m.kind == kind and not unmet(m, require)),
                       key=lambda m: m.id)  # fmt: skip
        return sorted(found, key=PREFER[prefer]) if prefer else found


def select(
    require: Mapping[str, Any] | None = None,
    *,
    kind: Kind = "chat",
    prefer: str | None = None,
    registry: Registry | None = None,
) -> list[ModelConfig]:
    """The models of `kind` that meet `require`, best first; empty when none does.

    mk.select({"features": ["camera angle"]}, kind="image")   # every listed feature declared
    """
    return (registry or load()).matching(require, prefer, kind)


def default_paths() -> list[Path]:
    """User and project registry files, in merge order (they may not exist)."""
    return [Path.home() / ".config" / "hone" / "models.toml", Path.cwd() / "hone-models.toml"]


def load(paths: Iterable[str | Path] | str | Path | None = None) -> Registry:
    """Load the packaged defaults, then user and project files if present, then `paths` (must exist)."""
    merged = _with_workflow_paths(_packaged(), Path(str(resources.files("hone_models"))) / "data")
    if isinstance(paths, str | Path):
        paths = [paths]
    explicit = [Path(p) for p in paths or ()]
    for path in explicit:
        if not path.is_file():
            raise ConfigError(f"registry file {path} does not exist")
    for path in [p for p in default_paths() if p.is_file()] + explicit:
        merged = merge(merged, _read_models(path))
    return Registry({mid: _parse(mid, raw) for mid, raw in merged.items()})


def _packaged() -> dict[str, Any]:
    """The packaged catalog: one file per kind in `data/models/` (change 0015 §3b); an id is in one file."""
    merged: dict[str, Any] = {}
    folder = resources.files("hone_models").joinpath("data", "models")
    for item in sorted(folder.iterdir(), key=lambda i: i.name):
        if item.name.endswith(".toml"):
            models = tomllib.loads(item.read_text(encoding="utf-8")).get("models", {})
            twice = sorted(set(models) & set(merged))
            if twice:
                raise ConfigError(f"packaged registry: {twice} declared again in data/models/{item.name}")
            merged.update(models)
    return merged


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
    return _with_workflow_paths(data.get("models", {}), path.parent)


def _with_workflow_paths(models: dict[str, Any], folder: Path) -> dict[str, Any]:
    """`models` with each relative `workflow` made absolute against the folder of the file declaring it."""
    for model_id, entry in models.items():
        raw: dict[str, Any] = entry if isinstance(entry, dict) else {}  # pyright: ignore[reportUnknownVariableType]
        if isinstance(raw.get("workflow"), str):  # an absolute path stays as is
            models[model_id] = {**raw, "workflow": str(folder / Path(raw["workflow"]).expanduser())}
    return models


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
