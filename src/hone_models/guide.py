"""Model guides (change 0015 §3a): what each model can take, for whoever writes its prompt.

    g = mk.guide("qwen-image-edit-2511")
    g.summary, g.prompt, g.features[0].name, g.features[0].examples, g.source
    g.inputs            # every input the model takes: common ones, its own, prompt inputs with their choices
    g.as_text()         # one block of plain text, for a person or for an LLM prompt
    g.as_dict()         # the JSON form

A guide is documentation, never used by the call itself. An entry without a `guide` table still has one,
built from its inputs and capabilities, with `summary = None`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

from . import catalog
from ._registry_shapes import GuideFeature
from .formats import prompt_input_names
from .providers import MEDIA
from .providers.common import COMMON_INPUTS, listed_inputs
from .registry import MEDIA_KINDS, ModelConfig, Registry, load

# What the shared inputs mean (change 0015 §1); an entry's own `guide.inputs` note wins.
COMMON_NOTES = {
    "negative": "what the output should not have",
    "size": "'WxH', e.g. '1024x1024'",
    "references": "image files whose content the result keeps (a list of paths)",
    "image": "a start frame (an image file)",
    "source": "an audio or video file to transform",
    "strength": "0-1: how far the result moves away from `source`",
    "lyrics": "common format: section tags ([verse], [chorus], ...) on their own line, one sung line a line",
    "duration_s": "seconds of output",
    "n": "how many outputs",
    "steps": "sampling steps",
}


@dataclass(frozen=True, slots=True)
class GuideInput:
    """One input a model takes: where it comes from (`common`, the model's `own`, or a `prompt` input
    written into the prompt), a note, and for a prompt input its choices (value -> phrase)."""

    name: str
    origin: Literal["common", "own", "prompt"]
    note: str | None = None
    choices: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ModelGuide:
    """What a model can take: its guide, every input it accepts, its limits, licence and install state."""

    id: str
    kind: str
    summary: str | None
    prompt: str | None
    inputs: dict[str, GuideInput]
    features: list[GuideFeature]
    source: str | None
    checked: date | None
    license: str | None
    commercial_use: bool | None
    sizes: list[str] | None
    durations_s: list[float] | None
    max_duration_s: float | None
    max_references: int | None
    lyrics_format: str | None
    installed: str
    install: list[str] = field(default_factory=list[str])

    def as_dict(self) -> dict[str, Any]:
        """The JSON form (dates as ISO strings)."""
        data: dict[str, Any] = {f.name: getattr(self, f.name) for f in fields(self)}
        data["inputs"] = {
            n: {"origin": i.origin, "note": i.note, "choices": i.choices} for n, i in self.inputs.items()
        }
        data["features"] = [f.model_dump() for f in self.features]
        data["checked"] = self.checked.isoformat() if self.checked else None
        return data

    def as_text(self) -> str:
        """One block of plain text, for a person or for an LLM prompt."""
        lines = [f"{self.id} ({self.kind})", *([self.summary] if self.summary else [])]
        if self.prompt:
            lines.append(f"Prompt: {self.prompt}")
        if self.inputs:
            lines.append("Inputs:")
            lines += [f"  {name}: {_input_text(item)}" for name, item in sorted(self.inputs.items())]
        if self.features:
            lines.append("Features:")
            for feature in self.features:
                lines += _feature_text(feature)
        lines += _limits_text(self)
        how = f" (install: hone-models models install {self.id})"
        lines.append(
            f"Installed: {self.installed}" + ("" if self.installed == "yes" or not self.install else how)
        )
        if self.source:
            lines.append(f"Source: {self.source}" + (f" (checked {self.checked})" if self.checked else ""))
        return "\n".join(lines)


def guide(model_id: str, *, registry: Registry | None = None) -> ModelGuide:
    """The `ModelGuide` of a registered (or ad-hoc) model, with its install state on this machine."""
    return build((registry or load()).get(model_id))


def build(cfg: ModelConfig, installed: str | None = None) -> ModelGuide:
    """The guide of one entry; `installed` is checked on this machine unless given."""
    written = cfg.guide
    caps = cfg.capabilities
    return ModelGuide(
        id=cfg.id, kind=cfg.kind, summary=written.summary if written else None,
        prompt=written.prompt if written else None, inputs=accepted_inputs(cfg),
        features=list(written.features) if written else [], source=written.source if written else None,
        checked=written.checked if written else None, license=caps.license,
        commercial_use=caps.commercial_use, sizes=caps.sizes, durations_s=caps.durations_s,
        max_duration_s=caps.max_duration_s, max_references=caps.max_references,
        lyrics_format=cfg.lyrics_format, installed=installed or catalog.installed(cfg),
        install=catalog.install_commands(cfg),
    )  # fmt: skip


def accepted_inputs(cfg: ModelConfig) -> dict[str, GuideInput]:
    """Every named input the entry takes (besides `prompt` and `seed`), with its note."""
    notes = cfg.guide.inputs if cfg.guide else {}
    prompt_inputs = cfg.prompt_inputs or {}
    found: dict[str, GuideInput] = {}
    for name in sorted(_names(cfg)):
        if name in prompt_inputs:
            found[name] = GuideInput(name, "prompt", notes.get(name), dict(prompt_inputs[name].choices))
        else:
            origin = "common" if name in COMMON_INPUTS else "own"
            note = notes.get(name) or COMMON_NOTES.get(name)
            if name == "lyrics" and cfg.lyrics_format and name not in notes:
                note = f"{note}; converted to the model's '{cfg.lyrics_format}' form"
            found[name] = GuideInput(name, origin, note)
    return found


def stale(registry: Registry, days: int, today: date | None = None) -> list[tuple[str, date | None]]:
    """The entries whose guide was never checked, or not within `days`, oldest first."""
    today = today or datetime.now(UTC).date()
    limit = today - timedelta(days=days)
    found = [(m.id, m.guide.checked if m.guide else None) for m in registry.models.values()]
    old = [(i, c) for i, c in found if c is None or c < limit]
    return sorted(old, key=lambda row: (row[1] or date.min, row[0]))


def _names(cfg: ModelConfig) -> set[str]:
    names: set[str] = set()
    if cfg.kind in MEDIA_KINDS:
        provider = MEDIA.get(cfg.provider)
        names = provider.accepted(cfg) if provider else listed_inputs(cfg)
    elif isinstance(cfg.inputs, list):
        names = set(cfg.inputs)
    return names | prompt_input_names(cfg)


def _input_text(item: GuideInput) -> str:
    text = item.note or item.origin
    if item.choices:
        text += f" (choices: {', '.join(sorted(item.choices))})"
    return text


def _feature_text(feature: GuideFeature) -> list[str]:
    head = f"  {feature.name}" + (f" (input {feature.input})" if feature.input else "")
    lines = [head + (f": {feature.how}" if feature.how else "")]
    lines += [f"    e.g. {example}" for example in feature.examples]
    return lines + ([f"    source: {feature.source}"] if feature.source else [])


def _limits_text(g: ModelGuide) -> list[str]:
    limits = [
        f"sizes {', '.join(g.sizes)}" if g.sizes else None,
        f"durations {', '.join(f'{d:g}' for d in g.durations_s)} s" if g.durations_s else None,
        f"at most {g.max_duration_s:g} s" if g.max_duration_s is not None else None,
        f"at most {g.max_references} references" if g.max_references is not None else None,
    ]
    shown = [limit for limit in limits if limit]
    lines = [f"Limits: {'; '.join(shown)}"] if shown else []
    if g.license or g.commercial_use is not None:
        use = {True: "allowed", False: "not allowed", None: "not stated"}[g.commercial_use]
        lines.append(f"License: {g.license or 'not stated'} (commercial use: {use})")
    return lines
