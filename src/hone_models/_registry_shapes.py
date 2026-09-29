"""The tables inside a registry entry: capabilities and price, and those a generation entry adds (change
0015 §3, §3a, §3b): how its named inputs map onto a ComfyUI workflow, prompt inputs, the model guide and
where the model comes from.

These are shapes only: `registry.py` checks them when an entry is loaded; the clients and the providers
give them their meaning.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Price(BaseModel):
    """USD per million tokens; for generation, per image or per second of output (a flat estimate)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    per_image: float = 0.0
    per_output_second: float = 0.0


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
    # generation and transcription (change 0015)
    max_references: int | None = None
    sizes: list[str] | None = None  # "WxH"
    max_duration_s: float | None = None
    durations_s: list[float] | None = None
    word_timestamps: bool | None = None
    commercial_use: bool | None = None  # information only: never blocks a call
    features: list[str] | None = None  # the names of the guide's features


_NODE_PATH = re.compile(r"^[^.\s]+\.[^.\s]+$")  # "<node id>.<input name>"


def check_node_path(path: str) -> str:
    """`path` if it reads `"<node id>.<input>"`, else a `ValueError` saying so."""
    if not _NODE_PATH.match(path):
        raise ValueError(f"{path!r} is not a workflow path '<node id>.<input>'")
    return path


class NodeInput(BaseModel):
    """One workflow input a named input fills; `per_second` / `add` convert seconds to frames
    (`round(seconds * per_second + add)`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    per_second: float | None = None
    add: float = 0.0

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        return check_node_path(value)


# A named input of a `comfyui` entry fills one or more workflow inputs.
NodeTarget = str | NodeInput
NodeMapping = NodeTarget | list[NodeTarget]


def node_targets(mapping: NodeMapping) -> list[NodeInput]:
    """Every workflow input one named input fills, as `NodeInput`s."""
    items = mapping if isinstance(mapping, list) else [mapping]
    return [NodeInput(path=item) if isinstance(item, str) else item for item in items]


def check_node_mapping(mapping: dict[str, NodeMapping]) -> dict[str, NodeMapping]:
    """Every string path of a `comfyui` input table checked (`NodeInput` checks its own)."""
    for name, value in mapping.items():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str):
                try:
                    check_node_path(item)
                except ValueError as exc:
                    raise ValueError(f"inputs.{name}: {exc}") from None
    return mapping


class PromptInput(BaseModel):
    """A named input written into the prompt as the phrase its choice maps to (§3a)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    place: Literal["append", "prepend"] = "append"
    choices: dict[str, str] = Field(min_length=1)


class GuideFeature(BaseModel):
    """A special ability of a model: how to use it, a few examples and where the full list lives."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    how: str | None = None
    input: str | None = None
    examples: list[str] = []
    source: str | None = None


class Guide(BaseModel):
    """What a model can take, for whoever writes its prompt (§3a); never used by the call itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    summary: str | None = None
    prompt: str | None = None
    inputs: dict[str, str] = {}
    features: list[GuideFeature] = []
    source: str | None = None
    checked: date | None = None


class InstallFile(BaseModel):
    """A Hugging Face file and the ComfyUI model folder it goes to; without `file`, the whole repository
    goes into that folder (`to` is then the model's own folder, e.g. `heartmula/HeartCodec-oss`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    repo: str
    file: str | None = None
    to: str


class Install(BaseModel):
    """Where a model comes from and what to fetch (§3b): an Ollama name, Hugging Face files for ComfyUI,
    a whole Hugging Face repository for the HF cache (`hf`), or a project to clone and set up (`repo`,
    `setup`, its folder variable `dir_env` and a `check` command)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str | None = None
    ollama: str | None = None
    files: list[InstallFile] = []
    hf: str | None = None
    repo: str | None = None
    setup: list[str] = []
    dir_env: str | None = None
    check: str | None = None
    size_gb: float | None = None
    tier: Literal[1, 2, 3] | None = None
    note: str | None = None


def feature_names(guide: Any) -> list[str]:
    """The feature names of a raw or parsed `guide` table (empty without one)."""
    if isinstance(guide, Guide):
        return [f.name for f in guide.features]
    raw = cast(dict[str, Any], guide) if isinstance(guide, dict) else {}
    features = cast(list[Any], raw.get("features") or [])
    return [str(cast(dict[str, Any], f)["name"]) if isinstance(f, dict) else str(f.name) for f in features]


def shape_errors(provider: str, raw: dict[str, Any]) -> list[str]:
    """What is wrong with the provider-specific keys of an entry (empty when nothing is)."""
    inputs, errors = raw.get("inputs"), list[str]()
    if provider == "comfyui":
        if inputs is not None and not isinstance(inputs, dict):
            errors.append("inputs of a comfyui entry is a table of workflow paths")
        errors += [f"{key} is for command entries" for key in ("command", "cwd", "env") if key in raw]
    else:
        if isinstance(inputs, dict):
            errors.append("inputs is a list of input names (a table of paths is for comfyui entries)")
        errors += [f"{key} is for comfyui entries" for key in ("workflow", "outputs") if key in raw]
        if provider != "command":
            errors += [f"{key} is for command entries" for key in ("command", "cwd", "env") if key in raw]
    return errors
