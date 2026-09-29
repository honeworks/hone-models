"""`.env` files (change 0017): read once per process when the registry loads; never overwrite a variable.

    KEY=value            # an unquoted value ends at " #"
    export KEY="a # b"   # quotes (single or double) are taken literally, no escapes

Values are never logged, and an error names the file and line, never the line's text.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .errors import ConfigError

KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_READ: set[Path] = set()  # files already applied in this process


def load_env_file(path: Path) -> dict[str, str]:
    """The variables in a `.env` file, in order; a malformed line is a `ConfigError` (file and line)."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"env file {path} is not UTF-8 text") from exc
    found: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pair = _pair(line.removeprefix("export ").lstrip())
        if pair is None:
            raise ConfigError(f"env file {path} line {number}: expected KEY=VALUE (quotes must be closed)")
        found[pair[0]] = pair[1]
    return found


def _pair(line: str) -> tuple[str, str] | None:
    key, sep, value = line.partition("=")
    key, value = key.strip(), value.strip()
    if not sep or not KEY.fullmatch(key):
        return None
    if value[:1] in ("'", '"'):
        quote = value[0]
        if len(value) < 2 or not value.endswith(quote):
            return None
        return key, value[1:-1]
    return key, value.split(" #", 1)[0].rstrip()


def apply_env_files() -> None:
    """Export the variables of `$HONE_ENV_FILE` (must exist) and then `./.env` that are not set yet."""
    paths: list[Path] = []
    named = os.environ.get("HONE_ENV_FILE")
    if named:
        if not Path(named).is_file():
            raise ConfigError(f"HONE_ENV_FILE names {named}, which does not exist")
        paths.append(Path(named))
    if Path(".env").is_file():
        paths.append(Path(".env"))
    for path in paths:
        resolved = path.resolve()
        if resolved in _READ:
            continue
        for key, value in load_env_file(path).items():
            os.environ.setdefault(key, value)
        _READ.add(resolved)
