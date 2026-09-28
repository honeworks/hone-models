"""Prompts built from named, versioned sections, so records can say which part of a prompt caused what.

    prompt = mk.Prompt("song_ideas", "3", sections={
        "system": "You write song pitches.",
        "rules": "Exactly 12 ideas, JSONL.",
        "brief": "Theme: {theme}",
    }, variables={"theme": "sea shanties"})
    llm.complete(prompt)

Rendering: system sections (in `system_sections`) form the system message, the others the user message,
each joined by a blank line, in the given order. `prompt.text` is all of them joined the same way (system
first); every section's character span `[start, end)` points into that text.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

SEPARATOR = "\n\n"
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True, slots=True)
class Section:
    """A section's text with its own version (defaults to the prompt's version)."""

    text: str
    version: str | None = None


def fill(text: str, variables: Mapping[str, Any]) -> str:
    """Replace `{name}` for names in `variables`; other braces (JSON examples) are left alone."""
    return _PLACEHOLDER.sub(
        lambda m: str(variables[m.group(1)]) if m.group(1) in variables else m.group(0), text
    )


@dataclass(frozen=True)
class Prompt:
    """A prompt template instance: id, version, ordered sections and variables."""

    template_id: str
    version: str
    sections: Mapping[str, str | Section]
    variables: Mapping[str, Any] | None = None
    system_sections: tuple[str, ...] = ("system",)
    _parts: list[tuple[str, str, str, str]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Own copies: later changes to the caller's dicts must not make `text` and records disagree.
        object.__setattr__(self, "sections", dict(self.sections))
        object.__setattr__(self, "variables", dict(self.variables) if self.variables is not None else None)
        parts: list[tuple[str, str, str, str]] = []  # (id, version, role, rendered text)
        for sid, value in self.sections.items():
            section = value if isinstance(value, Section) else Section(value)
            role = "system" if sid in self.system_sections else "user"
            parts.append(
                (sid, section.version or self.version, role, fill(section.text, self.variables or {}))
            )
        parts.sort(key=lambda p: p[2] != "system")  # stable: system sections first, order kept
        object.__setattr__(self, "_parts", parts)

    @property
    def text(self) -> str:
        """The full rendered text (system sections first)."""
        return SEPARATOR.join(p[3] for p in self._parts)

    def messages(self) -> list[dict[str, Any]]:
        """Chat messages: one system message (if any system section) and one user message."""
        out: list[dict[str, Any]] = []
        for role in ("system", "user"):
            texts = [p[3] for p in self._parts if p[2] == role]
            if texts:
                out.append({"role": role, "content": SEPARATOR.join(texts)})
        return out

    def section_records(self) -> list[dict[str, Any]]:
        """`hone.models.prompt.sections`: id, version, role, character span in `text`, sha256."""
        records: list[dict[str, Any]] = []
        start = 0
        for sid, version, role, text in self._parts:
            end = start + len(text)
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            records.append(
                {"id": sid, "version": version, "role": role, "start": start, "end": end, "sha256": sha}
            )
            start = end + len(SEPARATOR)
        return records

    def attributes(self) -> dict[str, Any]:
        """Span attributes for this prompt (design/current.md §8.3)."""
        return {
            "hone.models.prompt.template_id": self.template_id,
            "hone.models.prompt.template_version": self.version,
            "hone.models.prompt.sections": self.section_records(),
            "hone.models.prompt.variables": dict(self.variables or {}),
        }
