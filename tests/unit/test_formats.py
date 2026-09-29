"""The common lyrics format and its converters, and prompt inputs (hone_models.formats)."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hone_models import formats
from hone_models.errors import ConfigError
from hone_models.registry import ModelConfig

# The English example of SongGeneration's README, written in the common format.
README = """[intro-medium]
[verse]
Trails wind through the forest
Trees stand tall and honest
[chorus]
Forest is the sanctuary where the promise does fondest
[inst-medium]
[outro-medium]"""


def test_sections_parse_tags_lines_and_untagged_start() -> None:
    text = "first line\n\n[Verse] \n  one \n\n[CHORUS]\ntwo\nthree\n[outro]"
    assert formats.sections(text) == [
        (None, ["first line"]), ("verse", ["one"]), ("chorus", ["two", "three"]), ("outro", []),
    ]  # fmt: skip


def test_sections_format_is_passed_as_is() -> None:
    assert formats.convert_lyrics(README, "sections") == README


def test_plain_removes_tags() -> None:
    assert formats.to_plain("[verse]\na\nb\n[inst]\n[chorus]\nc") == "a\nb\n\nc"


def test_levo_matches_the_readme_example() -> None:
    assert formats.to_levo(README) == (
        "[intro-medium] ; [verse] Trails wind through the forest. Trees stand tall and honest. ; "
        "[chorus] Forest is the sanctuary where the promise does fondest. ; [inst-medium] ; [outro-medium]"
    )


def test_levo_maps_common_tags_and_cleans_punctuation() -> None:
    chinese = "\u82b1\u6735\u7efd\u653e\uff0c\n\u968f\u98ce\u8f7b\u821e\u52a8\u3002"  # full-width , and .
    text = f"untagged start;\n[pre-chorus]\nWait for it!\n[intro]\n[chorus]\n{chinese}\n[inst]\n[outro]"
    assert formats.to_levo(text) == (
        "[verse] untagged start. ; [verse] Wait for it. ; [intro-short] ; "
        "[chorus] \u82b1\u6735\u7efd\u653e.\u968f\u98ce\u8f7b\u821e\u52a8 ; [inst-short] ; [outro-short]"
    )  # Chinese lines: joined with '.', no final period (LeVo's README)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[verse]\n[chorus]\nla", r"\[verse\] has no lines"),
        ("[verse]\n...\n", r"\[verse\] has no lines"),
        ("[intro]\nhello\n[verse]\nla", r"\[intro-short\] is instrumental"),
        ("[hook]\nla", r"unknown section tag \[hook\]"),
        ("[intro]\n[outro]", "at least one sung section"),
    ],
)
def test_levo_refuses_what_songgeneration_cannot_sing(text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        formats.to_levo(text)


def test_unknown_lyrics_format() -> None:
    with pytest.raises(ConfigError, match="unknown lyrics_format 'abc'"):
        formats.convert_lyrics("x", "abc")


LINE = st.text(alphabet=st.characters(categories=["L"]), min_size=1, max_size=12)
SECTION = st.tuples(
    st.sampled_from(["verse", "chorus", "bridge", "pre-chorus"]), st.lists(LINE, min_size=1, max_size=4)
)


@given(st.lists(SECTION, min_size=1, max_size=6))
def test_converters_keep_every_sung_line(song: list[tuple[str, list[str]]]) -> None:
    text = "\n".join(f"[{tag}]\n" + "\n".join(lines) for tag, lines in song)
    assert formats.sections(formats.to_sections(text)) == [(tag, lines) for tag, lines in song]
    assert formats.to_plain(text).split() == [line for _, lines in song for line in lines]
    levo = formats.to_levo(text).split(" ; ")
    assert len(levo) == len(song)
    for part, (_, lines) in zip(levo, song, strict=True):
        assert all(line in part for line in lines)


def entry(**extra: object) -> ModelConfig:
    return ModelConfig.model_validate({"id": "m", "provider": "comfyui", "kind": "image", **extra})


def test_prompt_inputs_append_and_prepend() -> None:
    cfg = entry(
        prompt_inputs={
            "angle": {"choices": {"left": "LEFT PHRASE"}},
            "style": {"place": "prepend", "choices": {"ink": "ink drawing"}},
        }
    )
    assert formats.prompt_input_names(cfg) == {"angle", "style"}
    prompt, rest = formats.apply(cfg, "a girl.", {"angle": "left", "style": "ink", "size": "1x1"})
    assert prompt == "ink drawing, a girl. LEFT PHRASE"  # a comma unless a sentence already ended
    assert rest == {"size": "1x1"}  # prompt inputs never reach the provider
    assert formats.apply(cfg, "", {"angle": "left"})[0] == "LEFT PHRASE"
    assert formats.apply(cfg, "unchanged", {})[0] == "unchanged"


def test_unknown_choice_lists_the_choices() -> None:
    cfg = entry(prompt_inputs={"angle": {"choices": {"left": "L", "right": "R"}}})
    with pytest.raises(ConfigError, match=r"angle='up' is not one of its choices \['left', 'right'\]"):
        formats.apply(cfg, "x", {"angle": "up"})


def test_apply_converts_lyrics_by_the_entry_format() -> None:
    levo = entry(kind="music", lyrics_format="levo")
    assert formats.apply(levo, "p", {"lyrics": "[verse]\nla"})[1] == {"lyrics": "[verse] la."}
    assert formats.apply(entry(kind="music"), "p", {"lyrics": "[verse]\nla"})[1] == {"lyrics": "[verse]\nla"}
