"""Decision questions and answers in the shape of design/current.md §6.

Questions are plain dicts (`mk.YesNo`, `mk.Choice` and `mk.ScoreQ` are dicts with a constructor), so any
`{"type": ..., "instructions": ...}` mapping works too. Answers are plain dicts with every key of the
port's `Answer`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import ConfigError

QUESTION_TYPES = ("yes_no", "choice", "score")


class YesNo(dict[str, Any]):
    """A yes/no question; the answer's `value` is the probability of "yes"."""

    def __init__(self, instructions: str) -> None:
        super().__init__(type="yes_no", instructions=instructions)


class Choice(dict[str, Any]):
    """Pick one of `options`; the answer's `choice` is the option, `value` its probability."""

    def __init__(self, options: Sequence[str], instructions: str) -> None:
        super().__init__(type="choice", instructions=instructions, options=list(options))


class ScoreQ(dict[str, Any]):
    """A score on `scale`; the answer's `raw` is on the scale, `value` normalized to 0-1.

    `anchors` describe points of the scale, e.g. `{1: "flat", 5: "instantly singable"}`.
    """

    def __init__(
        self,
        instructions: str,
        scale: tuple[float, float] = (1, 5),
        anchors: Mapping[Any, str] | None = None,
    ) -> None:
        super().__init__(type="score", instructions=instructions, scale=list(scale))
        if anchors:
            self["anchors"] = {str(k): v for k, v in anchors.items()}


def check_questions(questions: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validated plain-dict copies of `questions` (score scale defaults to [1, 5]).

    Raises `ConfigError` for an unknown type, missing instructions, too few options or a bad scale.
    """
    if not questions:
        raise ConfigError("decide() needs at least one question, e.g. {'ok': mk.YesNo('Is it ok?')}")
    out: dict[str, dict[str, Any]] = {}
    for name, question in questions.items():
        q = dict(question)
        qtype = q.get("type")
        if qtype not in QUESTION_TYPES:
            raise ConfigError(f"question {name!r}: type must be one of {QUESTION_TYPES}, got {qtype!r}")
        if not q.get("instructions"):
            raise ConfigError(f"question {name!r}: 'instructions' is required")
        if qtype == "choice" and len(q.get("options") or []) < 2:
            raise ConfigError(f"question {name!r}: a choice needs at least two 'options'")
        if qtype == "score":
            low, high = q.setdefault("scale", [1, 5])
            if not low < high:
                raise ConfigError(f"question {name!r}: scale must be [low, high] with low < high")
        out[name] = q
    return out


def answer(qtype: str, **fields: Any) -> dict[str, Any]:
    """An `Answer` dict with every port key (unset keys are `None`, `calibrated` is `False`)."""
    base: dict[str, Any] = {
        "type": qtype,
        "value": None,
        "choice": None,
        "probabilities": None,
        "raw": None,
        "confidence": None,
        "calibrated": False,
        "rationale": None,
        "error": None,
    }
    return base | fields


def failed(qtype: str, error: str) -> dict[str, Any]:
    """The answer for a question that could not be answered: `value` is `None`, `error` says why."""
    return answer(qtype, error=error)


def probability(value: Any) -> float:
    """`value` as a float in [0, 1]; `ValueError` otherwise (NaN included)."""
    p = float(value)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability {value!r} is outside [0, 1]")
    return p


def normalized_score(raw: float, scale: Sequence[float]) -> float:
    """`(raw - low) / (high - low)`; `ValueError` when `raw` is outside the scale."""
    low, high = scale
    if not low <= raw <= high:
        raise ValueError(f"score {raw} is outside the scale [{low}, {high}]")
    return (raw - low) / (high - low)
