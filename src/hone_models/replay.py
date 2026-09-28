"""Replay a recorded chat call with changes (design/current.md §8.6).

    span = mk.records.read_spans(".hone/models/spans.db")[-1]
    new = mk.replay.Replayer().replay_call(span, {"prompt.sections": {"format_example": None}})
    new["attributes"]["hone.models.replay_of"] == span["span_id"]

The request is rebuilt from the span (it needs content capture): messages, params and schema are the
recorded ones. Overrides: `"messages"` (full replacement), `"prompt.sections"` (`{id: new_text | None}`,
`None` removes the section; texts are used as given, without variables), `"model"`, `"params"` (merged).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ._tracing import extra_attributes
from .errors import ConfigError
from .ports import RecordSink
from .prompt import SEPARATOR, Prompt, Section
from .records import MemorySink, default_sink
from .registry import ADHOC, Registry
from .text import text

OVERRIDE_KEYS = ("prompt.sections", "model", "params", "messages")
GEN_AI_PARAMS = ("temperature", "top_p", "max_tokens", "seed")


class Replayer:
    """Replays recorded `hone.models.chat` spans; new spans go to `sink` (default: the local store)."""

    def __init__(self, sink: RecordSink | None = None, registry: Registry | None = None) -> None:
        self.sink = sink or default_sink()
        self.registry = registry

    def replay_call(
        self,
        call_span: Mapping[str, Any],
        overrides: Mapping[str, Any],
        *,
        trace: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Call the model again with `overrides` applied; returns the new span (with `replay_of`)."""
        unknown = set(overrides) - set(OVERRIDE_KEYS)
        if unknown:
            raise ConfigError(f"unknown replay overrides {sorted(unknown)}; use {list(OVERRIDE_KEYS)}")
        if "messages" in overrides and "prompt.sections" in overrides:
            raise ConfigError("give 'messages' or 'prompt.sections' as a replay override, not both")
        attrs: Mapping[str, Any] = call_span["attributes"]
        messages = (
            overrides["messages"] if "messages" in overrides else recorded(attrs, "gen_ai.input.messages")
        )
        request = messages
        if "prompt.sections" in overrides:
            request = with_sections(attrs, messages, overrides["prompt.sections"])
        params = {**recorded_params(attrs), **overrides.get("params", {})}
        schema = attrs.get("hone.models.structured.schema")
        model_id = overrides.get("model") or recorded_model(attrs)

        # The new span is returned and stored in `sink`; it follows the sink's content-capture setting.
        captured = MemorySink(capture_content=getattr(self.sink, "capture_content", None))
        llm = text(model_id, registry=self.registry, sink=captured)
        try:
            with extra_attributes({"hone.models.replay_of": call_span["span_id"]}):
                llm.complete(request, schema=schema, trace=trace, **params)
        finally:
            for span in captured.spans:
                self.sink.emit(span)
        return captured.spans[-1]


def decoded(value: Any) -> Any:
    """A list / object attribute stored as a JSON string (the OpenTelemetry encoding), decoded."""
    if isinstance(value, str) and value[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def recorded(attrs: Mapping[str, Any], key: str) -> Any:
    """A content attribute of the span; `ConfigError` when it was not captured."""
    value: Any = decoded(attrs.get(key))
    hashed = isinstance(value, dict) and value.keys() == {"sha256", "len"}
    if value is None or hashed:
        raise ConfigError(f"the span has no {key}; replay needs spans recorded with content capture on")
    return value  # pyright: ignore[reportUnknownVariableType]


def recorded_params(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """All recorded call params, or the `gen_ai.request.*` ones for spans from other recorders."""
    if "hone.models.request.params" in attrs:
        return dict(recorded(attrs, "hone.models.request.params"))
    return {k: attrs[f"gen_ai.request.{k}"] for k in GEN_AI_PARAMS if f"gen_ai.request.{k}" in attrs}


def recorded_model(attrs: Mapping[str, Any]) -> str:
    """The registry id, else an ad-hoc `provider:model` id from the `gen_ai.*` attributes."""
    if "hone.models.model_id" in attrs:
        return str(attrs["hone.models.model_id"])
    provider, model = attrs.get("gen_ai.provider.name"), attrs.get("gen_ai.request.model")
    if provider in ADHOC and model:
        return f"{provider}:{model}"
    raise ConfigError("the span names no model; pass overrides={'model': '<registry id>'}")


def with_sections(
    attrs: Mapping[str, Any], messages: list[dict[str, Any]], changes: Mapping[str, str | None]
) -> Prompt:
    """The recorded prompt rebuilt from its section spans, with `changes` applied."""
    sections: list[dict[str, Any]] = decoded(attrs.get("hone.models.prompt.sections")) or []
    if not sections:
        raise ConfigError("the span has no hone.models.prompt.sections; it was not made from an mk.Prompt")
    missing = set(changes) - {s["id"] for s in sections}
    if missing:
        raise ConfigError(f"unknown sections {sorted(missing)}; the prompt has {[s['id'] for s in sections]}")
    rendered = SEPARATOR.join(str(m["content"]) for m in messages)
    new: dict[str, Section] = {}
    for s in sections:
        body = changes.get(s["id"], rendered[s["start"] : s["end"]])
        if body is not None:
            new[s["id"]] = Section(body, s["version"])
    system = tuple(s["id"] for s in sections if (s.get("role") or role_at(messages, s["start"])) == "system")
    template_id = attrs.get("hone.models.prompt.template_id", "")
    version = attrs.get("hone.models.prompt.template_version", "")
    return Prompt(template_id, version, new, system_sections=system)


def role_at(messages: list[dict[str, Any]], start: int) -> str:
    """The role of the message holding character `start` of the rendered prompt (for section records
    without a `role`, as other recorders write them)."""
    end = 0
    for message in messages:
        end += len(str(message["content"])) + len(SEPARATOR)
        if start < end:
            return str(message["role"])
    return "user"
