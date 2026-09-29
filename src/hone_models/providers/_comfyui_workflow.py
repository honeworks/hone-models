"""A `comfyui` entry's workflow: loading it, and filling a copy with one call's named inputs (0015 §2).

The entry's `inputs` table maps each named input to one or more `"<node id>.<input>"` paths; `size`
fills the `width` and `height` mappings; a `{ path, per_second, add }` mapping converts seconds to
frames; `references` maps to a list of slots, and a slot without a reference is removed with every link
to it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .._registry_shapes import NodeInput, node_targets
from ..errors import CapabilityError, ConfigError
from ..registry import ModelConfig

Workflow = dict[str, dict[str, Any]]
SIZE_TARGETS = ("width", "height")
NOT_INPUTS = ("prompt", "seed", *SIZE_TARGETS)  # filled from `prompt`, `seed=` and `size`


def accepted(cfg: ModelConfig) -> set[str]:
    """The named inputs the entry maps, `size` for `width` / `height`; `prompt` and `seed` are the call's
    own arguments."""
    mapping = cfg.inputs if isinstance(cfg.inputs, dict) else {}
    names = {name for name in mapping if name not in NOT_INPUTS}
    return names | ({"size"} if set(SIZE_TARGETS) & set(mapping) else set())


def load(cfg: ModelConfig) -> tuple[Workflow, str]:
    """The entry's API-format workflow and its SHA-256."""
    if not cfg.workflow:
        raise ConfigError(
            f"model {cfg.id!r} has no workflow yet: export its ComfyUI workflow with 'Export (API)' and "
            f"set `workflow` in its registry entry (then prove it with `hone-models models check {cfg.id}`)"
        )
    path = Path(cfg.workflow)
    try:
        raw = path.read_bytes()
        workflow = json.loads(raw)
    except OSError as exc:
        raise ConfigError(f"model {cfg.id!r}: cannot read its workflow {path}: {exc}") from exc
    except ValueError as exc:
        raise ConfigError(f"model {cfg.id!r}: workflow {path} is not JSON: {exc}") from exc
    if not isinstance(workflow, dict) or not all(
        isinstance(n, dict) and "class_type" in n and "inputs" in n
        for n in workflow.values()  # pyright: ignore[reportUnknownVariableType]
    ):
        raise ConfigError(
            f"model {cfg.id!r}: workflow {path} is not in ComfyUI's API format (use 'Export (API)')"
        )
    return workflow, hashlib.sha256(raw).hexdigest()  # pyright: ignore[reportUnknownVariableType]


def fill(workflow: Workflow, cfg: ModelConfig, values: dict[str, Any]) -> Workflow:
    """A copy of `workflow` with `values` (`prompt`, `seed` and the named inputs, files already replaced by
    their uploaded names) written into the mapped nodes."""
    mapping = cfg.inputs if isinstance(cfg.inputs, dict) else {}
    filled = copy.deepcopy(workflow)
    values = dict(values)
    if "size" in values and "size" not in mapping:
        width, height = str(values.pop("size")).split("x")
        values.update({"width": int(width), "height": int(height)})
    references: list[Any] | None = values.pop("references", None) if "references" in mapping else None
    for name, value in values.items():
        if name not in mapping:
            if name == "prompt" and value:
                raise ConfigError(f"model {cfg.id!r} maps no `prompt` input; add one to its `inputs` table")
            continue  # an unmapped seed, or a size part the workflow does not take
        for target in node_targets(mapping[name]):
            _set(filled, target, _converted(value, target), cfg, name)
    if "references" in mapping:
        _fill_slots(filled, node_targets(mapping["references"]), references or [], cfg)
    return filled


def _converted(value: Any, target: NodeInput) -> Any:
    if target.per_second is None:
        return value
    return round(float(value) * target.per_second + target.add)


def _set(workflow: Workflow, target: NodeInput, value: Any, cfg: ModelConfig, name: str) -> None:
    node, field = target.path.split(".", 1)
    where = f"model {cfg.id!r}: input {name!r} maps to {target.path!r}, but workflow {cfg.workflow}"
    if node not in workflow:
        raise ConfigError(f"{where} has no node {node!r}")
    if field not in workflow[node]["inputs"]:
        raise ConfigError(f"{where} has no input {field!r} on node {node!r} ({workflow[node]['class_type']})")
    workflow[node]["inputs"][field] = value


def _fill_slots(workflow: Workflow, slots: list[NodeInput], references: list[Any], cfg: ModelConfig) -> None:
    """One reference per slot, in order; slots left over are removed with the links to them."""
    if len(references) > len(slots):
        raise CapabilityError(f"model {cfg.id!r} has {len(slots)} reference slots, not {len(references)}")
    for slot, reference in zip(slots, references, strict=False):
        _set(workflow, slot, reference, cfg, "references")
    for slot in slots[len(references) :]:
        remove_node(workflow, slot.path.split(".", 1)[0])


def remove_node(workflow: Workflow, node: str) -> None:
    """Drop `node` and every input linked to it (`[node, output index]`); ComfyUI's own validation then
    says if a required input is missing."""
    workflow.pop(node, None)
    for other in workflow.values():
        inputs: dict[str, Any] = other["inputs"]
        for key, value in list(inputs.items()):
            if isinstance(value, list) and len(value) == 2 and str(value[0]) == node:  # pyright: ignore[reportUnknownArgumentType]
                del inputs[key]
