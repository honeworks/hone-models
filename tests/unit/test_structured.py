import contextlib
import json

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel

from hone_models.budget import estimate_tokens, output_budget, plan_context
from hone_models.errors import ValidationFailed
from hone_models.providers.common import ChatReply
from hone_models.registry import Capabilities, ModelConfig
from hone_models.structured import (
    parse_json,
    repair_json,
    run_structured,
    schema_dict,
    schema_errors,
    validate,
    with_schema_instruction,
)


class Item(BaseModel):
    name: str


def test_parse_json_clean_and_cleaned() -> None:
    assert parse_json('{"a": 1}') == ({"a": 1}, True)
    assert parse_json("```\n[1, 2]\n```") == ([1, 2], False)
    assert parse_json('Here you go: {"a": 1}') == ({"a": 1}, False)
    with pytest.raises(ValueError, match="Expecting value"):
        parse_json("no json here")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1, "b": "tru', {"a": 1}),
        ('[{"x": 1}, {"x": 2}, {"x"', [{"x": 1}, {"x": 2}]),
        ('{"a": {"b": [1, 2', {"a": {"b": [1]}}),
        ('{"s": "a, \\"quoted\\" b", "t": [', {"s": 'a, "quoted" b'}),
        ('```json\n{"a": [1, 2, 3]', {"a": [1, 2, 3]}),
        ('{"a": 1}', {"a": 1}),
    ],
)
def test_repair_json(text: str, expected: object) -> None:
    assert repair_json(text) == expected


def test_repair_json_gives_up() -> None:
    with pytest.raises(ValueError, match="could not repair"):
        repair_json("nothing")


@given(
    st.recursive(
        st.none() | st.booleans() | st.integers() | st.text(max_size=5),
        lambda c: st.lists(c, max_size=4) | st.dictionaries(st.text(max_size=4), c, max_size=4),
        max_leaves=12,
    )
)
def test_repair_never_crashes_on_any_prefix(value: object) -> None:
    text = json.dumps(value)
    for cut in range(1, len(text) + 1):
        with contextlib.suppress(ValueError):
            repair_json(text[:cut])
    if isinstance(value, dict | list):
        assert repair_json(text) == value


def test_validate_model_class_and_dict_schema() -> None:
    assert validate(Item, {"name": "x"}) == Item(name="x")
    with pytest.raises(ValidationFailed, match="name"):
        validate(Item, {})
    schema = {
        "type": "object",
        "required": ["n", "tags"],
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 5},
            "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
            "maybe": {"type": ["string", "null"]},
        },
    }
    assert validate(schema, {"n": 3, "tags": ["a"], "maybe": None}) == {"n": 3, "tags": ["a"], "maybe": None}
    assert schema_errors(schema, {"n": 9, "tags": ["c"]}) == [
        "$.n: 9 > maximum 5",
        "$.tags[0]: 'c' is not one of ['a', 'b']",
    ]
    assert schema_errors(schema, {"n": 0}) == ["$: missing required key 'tags'", "$.n: 0 < minimum 1"]
    assert schema_errors(schema, {"n": True, "tags": []}) == ["$.n: expected integer, got bool"]
    assert schema_errors({"type": "object"}, []) == ["$: expected object, got list"]
    with pytest.raises(ValidationFailed):
        validate(schema, {"n": 1})


def test_schema_dict_and_instruction() -> None:
    assert schema_dict(None) is None
    assert schema_dict(Item)["required"] == ["name"]  # type: ignore[index]
    msgs = with_schema_instruction([{"role": "user", "content": "hi"}], {"type": "object"})
    assert msgs[0]["role"] == "system"
    assert '{"type": "object"}' in msgs[0]["content"]
    msgs = with_schema_instruction(
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}], {}
    )
    assert msgs[0]["content"].startswith("Be brief.\n\nReply with only a JSON value")
    assert len(msgs) == 2


def test_budget() -> None:
    cfg = ModelConfig(
        id="m", provider="ollama", capabilities=Capabilities(max_input_tokens=8192, max_output_tokens=512)
    )
    assert estimate_tokens([{"content": "x" * 35}, {"content": [{"type": "text", "text": "y" * 7}]}]) == 12
    assert output_budget(cfg, {}) == 512
    assert output_budget(cfg, {"max_tokens": 100}) == 100
    assert output_budget(cfg, {"max_tokens": None}) == 512
    assert output_budget(ModelConfig(id="m", provider="ollama"), {}) == 2048
    assert plan_context(cfg, [{"content": "x" * 35}], 100) == (10, 2048, 0)  # rounded up to a 2048 step
    assert plan_context(cfg, [{"content": "x" * 35 * 700}], 1000) == (7000, 8192, 0)  # capped at the limit
    unknown = ModelConfig(id="m", provider="ollama")
    assert plan_context(unknown, [{"content": "x" * 35 * 1000}], 2048) == (10000, 14336, 0)


def test_length_finish_is_at_best_repaired() -> None:
    reply = ChatReply(text='{"name": "x"}', finish_reason="length", model="m")
    out = run_structured(lambda _: reply, [], Item, constrained=True)
    assert (out.path, out.parsed) == ("repaired", Item(name="x"))
    bad = ChatReply(text="{", finish_reason="length", model="m")
    out = run_structured(lambda _: bad, [], Item, constrained=True)
    assert out.path == "failed"
    assert out.error is not None
    assert "truncated" in out.error
