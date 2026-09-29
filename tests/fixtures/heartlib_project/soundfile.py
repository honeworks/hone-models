"""Fake soundfile: writes what the fake pipeline saw, as JSON."""

import json


def write(path: str, data: dict, rate: int, format: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({**data, "rate": rate, "format": format}, fh)
