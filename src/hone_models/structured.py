"""Structured output that never silently truncates (design/current.md §5).

1. budget (see `budget.py`): `ContextOverflow` before calling if prompt + output do not fit;
2. constrain when the model supports it (the provider sends the schema);
3. parse: strip markdown fences, `json.loads`;
4. validate: Pydantic `TypeAdapter` for a model class, a small JSON Schema check for a dict;
5. retry up to 2 times with the validation error sent back;
6. repair truncated JSON as a last resort (close structures after the last complete element);
7. finish reason `length` always makes the path at best `repaired`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from ._tracing import add_event
from .errors import ValidationFailed
from .providers.common import ChatReply, text_of

MAX_RETRIES = 2
_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)(?:```|$)", re.DOTALL)

Schema = type[BaseModel] | Mapping[str, Any]


def schema_dict(schema: Schema | None) -> dict[str, Any] | None:
    """JSON Schema for a Pydantic model class or a schema dict."""
    if schema is None:
        return None
    if isinstance(schema, type):
        return schema.model_json_schema()
    return dict(schema)


def with_schema_instruction(
    messages: list[dict[str, Any]], json_schema: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Add the schema to the system message (models follow it better even when constrained)."""
    note = "Reply with only a JSON value matching this JSON Schema, no other text:\n" + json.dumps(
        json_schema
    )
    if messages and messages[0].get("role") == "system":
        first = messages[0]
        return [{**first, "content": f"{text_of(first.get('content'))}\n\n{note}"}, *messages[1:]]
    return [{"role": "system", "content": note}, *messages]


def parse_json(text: str) -> tuple[Any, bool]:
    """Decode JSON from model output. Returns (value, clean) where clean means no cleanup was needed."""
    try:
        return json.loads(text), True
    except ValueError:
        pass
    value, _ = json.JSONDecoder().raw_decode(_strip(text))  # tolerates prose after the JSON
    return value, False


def _strip(text: str) -> str:
    match = _FENCE.search(text)
    if match:
        return match.group(1).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    return text[min(starts) :].strip() if starts else text.strip()


def repair_json(text: str) -> Any:
    """Best-effort decode of truncated JSON: cut after the last complete element, close open structures.

    Raises `ValueError` when nothing usable is left.
    """
    text = _strip(text)
    for pos, closing in reversed([(len(text), ""), *_cut_points(text)]):
        try:
            return json.loads(text[:pos] + closing)
        except ValueError:
            continue
    raise ValueError("could not repair truncated JSON")


def _cut_points(text: str) -> list[tuple[int, str]]:
    """Positions just after a complete element (before a comma, after a closing bracket), with the
    brackets still open there."""
    stack: list[str] = []
    cuts: list[tuple[int, str]] = []
    in_string = escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
            cuts.append((i + 1, "".join(reversed(stack))))
        elif ch == ",":
            cuts.append((i, "".join(reversed(stack))))
    return cuts


def validate(schema: Schema, value: Any) -> Any:
    """Validate `value`; return the model instance (class schema) or the value (dict schema)."""
    if isinstance(schema, type):
        try:
            return TypeAdapter(schema).validate_python(value)
        except PydanticValidationError as exc:  # message without the input values (they are model output)
            errors = exc.errors(include_input=False, include_url=False)
            raise ValidationFailed(
                "; ".join(f"{'.'.join(map(str, e['loc'])) or '$'}: {e['msg']}" for e in errors)
            ) from exc
    errors = schema_errors(schema, value)
    if errors:
        raise ValidationFailed("; ".join(errors))
    return value


_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def schema_errors(schema: Mapping[str, Any], value: Any, path: str = "$") -> list[str]:
    """A small JSON Schema check: type, enum, required, properties, items, minimum, maximum.

    Other keywords (`$ref`, `anyOf`, ...) are not checked; pass a Pydantic model for full validation.
    """
    wanted = schema.get("type")
    names: list[str] = [] if wanted is None else wanted if isinstance(wanted, list) else [wanted]  # pyright: ignore[reportUnknownVariableType]
    if names and not any(_TYPES.get(n, lambda _: True)(value) for n in names):
        return [f"{path}: expected {wanted}, got {type(value).__name__}"]
    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, dict):
        obj: dict[str, Any] = value  # pyright: ignore[reportUnknownVariableType]
        errors += [f"{path}: missing required key {k!r}" for k in schema.get("required", []) if k not in obj]
        for key, sub in schema.get("properties", {}).items():
            if key in obj:
                errors += schema_errors(sub, obj[key], f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        items: list[Any] = value  # pyright: ignore[reportUnknownVariableType]
        for i, item in enumerate(items):
            errors += schema_errors(schema["items"], item, f"{path}[{i}]")
    return errors + _bound_errors(schema, value, path)


def _bound_errors(schema: Mapping[str, Any], value: Any, path: str) -> list[str]:
    if not _TYPES["number"](value):
        return []
    errors: list[str] = []
    if "minimum" in schema and value < schema["minimum"]:
        errors.append(f"{path}: {value} < minimum {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    return errors


@dataclass(slots=True)
class Outcome:
    """Result of one call (`path` None) or of the structured pipeline. `usage` sums all attempts."""

    reply: ChatReply
    parsed: Any
    error: str | None
    path: str | None
    attempts: int
    usage: dict[str, int] = field(default_factory=dict[str, int])


def run_structured(
    send: Callable[[list[dict[str, Any]]], ChatReply],
    messages: list[dict[str, Any]],
    schema: Schema,
    *,
    constrained: bool,
) -> Outcome:
    """Call, parse, validate; retry with the error; repair as a last resort."""
    usage: dict[str, int] = {}
    error = ""
    for attempt in range(1, MAX_RETRIES + 2):
        reply = send(messages)
        for key, n in reply.usage.items():
            usage[key] = usage.get(key, 0) + n
        if reply.finish_reason == "length":
            return _repair(reply, schema, attempt, usage, "output truncated at max_tokens")
        try:
            value, clean = parse_json(reply.text)
            parsed = validate(schema, value)
        except (ValueError, ValidationFailed) as exc:
            error = f"{type(exc).__name__}: {exc}"
            add_event(
                "structured_retry", {"attempt": attempt, "reason": error[:500]}
            )  # no model output in it
            note = f"Your reply was not valid: {error}\nReply again with only the corrected JSON."
            messages = [
                *messages,
                {"role": "assistant", "content": reply.text},
                {"role": "user", "content": note},
            ]
            continue
        path = "retried" if attempt > 1 else ("constrained" if constrained and clean else "parsed")
        return Outcome(reply, parsed, None, path, attempt, usage)
    return _repair(reply, schema, MAX_RETRIES + 1, usage, error)  # pyright: ignore[reportPossiblyUnboundVariable]


def _repair(reply: ChatReply, schema: Schema, attempts: int, usage: dict[str, int], reason: str) -> Outcome:
    try:
        parsed = validate(schema, repair_json(reply.text))
    except (ValueError, ValidationFailed) as exc:
        error = f"structured output failed ({reason}); repair: {exc}"
        return Outcome(reply, None, error, "failed", attempts, usage)
    return Outcome(reply, parsed, None, "repaired", attempts, usage)
