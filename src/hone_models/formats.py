"""Common input formats (change 0015 §3a): what a caller writes once and hone-models converts per model.

Lyrics have one common format: section tags on their own line, then one sung line per line.

    [verse]
    Neon on the water
    [chorus]
    We run until the morning

An entry's `lyrics_format` names the converter: `sections` (as is), `levo` (SongGeneration's
`[verse] line. line. ; [chorus] ...`), `heartmula` (HeartMuLa's `[Verse]`, `[Prechorus]` markers, a blank
line between sections), `plain` (tags removed). `prompt_inputs` are named inputs written into
the prompt as the phrase the entry maps the chosen value to (`camera_angle="left_45"`).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .errors import ConfigError
from .registry import ModelConfig

SECTION_TAGS = ("intro", "verse", "pre-chorus", "chorus", "bridge", "inst", "outro")
_TAG = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
Section = tuple[str | None, list[str]]

# SongGeneration (LeVo) section labels, from its conf/vocab.yaml and README: instrumental sections have
# a length and no lines; the sung ones need lines.
LEVO_SUNG = ("verse", "chorus", "bridge")
LEVO_INSTRUMENTAL = (
    *(f"{part}-{length}" for part in ("intro", "inst", "outro") for length in ("short", "medium", "long")),
    "silence",
)
_HEARTMULA_TAGS = {"pre-chorus": "prechorus", "inst": "instrumental"}
_LEVO_TAGS = {"intro": "intro-short", "inst": "inst-short", "outro": "outro-short", "pre-chorus": "verse"}
# `;` separates LeVo's sections and `.` its lines; full-width punctuation is not allowed (README).
_LEVO_PUNCTUATION = str.maketrans({
    ";": ",", "\uff1b": ",", "\uff0c": ",", "\u3002": ".", "\uff01": "!", "\uff1f": "?", "\uff1a": ":",
})  # fmt: skip  # full-width ; , . ! ? : to their ASCII forms


def sections(text: str) -> list[Section]:
    """The lyrics as `(tag, lines)` pairs: tags lower-cased without brackets, blank lines dropped; lines
    before the first tag have the tag `None`."""
    found: list[Section] = []
    for raw in text.splitlines():
        line = raw.strip()
        tag = _TAG.match(line)
        if tag:
            found.append((tag.group(1).strip().lower(), []))
        elif line:
            if not found:
                found.append((None, []))
            found[-1][1].append(line)
    return found


def to_sections(text: str) -> str:
    """The common format as is (ACE-Step, MiniMax-Music3 and YuE2 read these tags)."""
    return text


def to_plain(text: str) -> str:
    """Only the sung lines: tags removed, a blank line between sections."""
    return "\n\n".join("\n".join(lines) for _, lines in sections(text) if lines)


def to_heartmula(text: str) -> str:
    """HeartMuLa's form (heartlib's README): `[Intro]`, `[Verse]`, `[Prechorus]`, `[Chorus]`, `[Bridge]`,
    `[Outro]` on their own line, a blank line between sections. `[pre-chorus]` becomes `[Prechorus]` and
    `[inst]` `[Instrumental]` (the ComfyUI node's marker); other tags are capitalised as they are."""
    parts: list[str] = []
    for tag, lines in sections(text):
        head = [f"[{_HEARTMULA_TAGS.get(tag, tag).capitalize()}]"] if tag else []
        parts.append("\n".join(head + lines))
    return "\n\n".join(parts)


def to_levo(text: str) -> str:
    """SongGeneration's form: `[intro-short] ; [verse] Line one. Line two. ; [chorus] ...`.

    `[intro]`, `[inst]` and `[outro]` become their `-short` forms, `[pre-chorus]` a `[verse]`, lines
    before any tag a `[verse]`. A sung section needs lines and an instrumental one takes none."""
    parts: list[str] = []
    sung = False
    for tag, lines in sections(text):
        label = _levo_label(tag or "verse")
        body = _levo_lines(lines)
        if label in LEVO_SUNG and not body:
            raise ConfigError(f"lyrics: [{tag}] has no lines; SongGeneration sings every [{label}]")
        if label not in LEVO_SUNG and body:
            raise ConfigError(f"lyrics: SongGeneration's [{label}] is instrumental: drop its lines")
        sung = sung or label in LEVO_SUNG
        parts.append(f"[{label}] {body}" if body else f"[{label}]")
    if not sung:
        raise ConfigError(f"lyrics: SongGeneration needs at least one sung section {list(LEVO_SUNG)}")
    return " ; ".join(parts)


def _levo_label(tag: str) -> str:
    label = _LEVO_TAGS.get(tag, tag)
    if label not in LEVO_SUNG + LEVO_INSTRUMENTAL:
        raise ConfigError(
            f"lyrics: unknown section tag [{tag}]; use {['[' + t + ']' for t in SECTION_TAGS]} "
            f"or SongGeneration's own {['[' + t + ']' for t in LEVO_INSTRUMENTAL]}"
        )
    return label


def _levo_lines(lines: list[str]) -> str:
    """Lines joined with `.`: English ends with a period, Chinese does not (LeVo's README)."""
    cleaned = [line.translate(_LEVO_PUNCTUATION).strip().rstrip(".,!?:").strip() for line in lines]
    cleaned = [line for line in cleaned if line]
    if not cleaned:
        return ""
    if _CJK.search(cleaned[-1][-1:]):
        return ".".join(cleaned)
    return ". ".join(cleaned) + "."


LYRICS: dict[str, Callable[[str], str]] = {
    "sections": to_sections, "levo": to_levo, "heartmula": to_heartmula, "plain": to_plain,
}  # fmt: skip


def convert_lyrics(text: str, lyrics_format: str) -> str:
    """`text` in the common format, converted with the converter `lyrics_format` names."""
    try:
        return LYRICS[lyrics_format](text)
    except KeyError:
        raise ConfigError(f"unknown lyrics_format {lyrics_format!r}; use one of {sorted(LYRICS)}") from None


def prompt_input_names(cfg: ModelConfig) -> set[str]:
    """The entry's prompt inputs: named inputs it takes that end up as words in the prompt."""
    return set(cfg.prompt_inputs or {})


def apply(cfg: ModelConfig, prompt: str, named: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The prompt with the prompt inputs written in, and the inputs the provider gets: prompt inputs
    removed, `lyrics` converted to the entry's `lyrics_format`."""
    rest = dict(named)
    for name, spec in (cfg.prompt_inputs or {}).items():
        if name not in rest:
            continue
        choice = rest.pop(name)
        phrase = spec.choices.get(str(choice))
        if phrase is None:
            raise ConfigError(
                f"model {cfg.id!r}: {name}={choice!r} is not one of its choices {sorted(spec.choices)}"
            )
        prompt = _join(prompt, phrase) if spec.place == "append" else _join(phrase, prompt)
    if cfg.lyrics_format and isinstance(rest.get("lyrics"), str):
        rest["lyrics"] = convert_lyrics(rest["lyrics"], cfg.lyrics_format)
    return prompt, rest


def _join(first: str, second: str) -> str:
    """Two prompt parts: a comma between them unless the first already ends a sentence or clause."""
    first, second = first.strip(), second.strip()
    if not first or not second:
        return first or second
    return f"{first} {second}" if first[-1] in ".!?,;:" else f"{first}, {second}"
