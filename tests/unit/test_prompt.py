import pytest

import hone_models as mk
from hone_models.prompt import fill


def test_fill_leaves_json_braces_alone() -> None:
    assert fill('Theme: {theme}. Example: {"title": "x"} {missing}', {"theme": "rain"}) == (
        'Theme: rain. Example: {"title": "x"} {missing}'
    )


def test_prompt_without_system_sections() -> None:
    prompt = mk.Prompt("t", "1", {"only": "Hello"})
    assert prompt.messages() == [{"role": "user", "content": "Hello"}]
    assert prompt.section_records()[0]["start"] == 0
    assert prompt.section_records()[0]["end"] == 5
    assert prompt.section_records()[0]["role"] == "user"


def test_prompt_is_immutable() -> None:
    prompt = mk.Prompt("t", "1", {"a": "x"})
    with pytest.raises(AttributeError):
        prompt.version = "2"  # type: ignore[misc]


def test_prompt_keeps_its_own_copy_of_sections_and_variables() -> None:
    sections = {"brief": "Theme: {theme}"}
    variables = {"theme": "rain"}
    prompt = mk.Prompt("t", "1", sections, variables=variables)
    sections["brief"] = "changed"
    variables["theme"] = "sea"
    assert prompt.sections == {"brief": "Theme: {theme}"}
    assert prompt.text == "Theme: rain"
    assert prompt.attributes()["hone.models.prompt.variables"] == {"theme": "rain"}
