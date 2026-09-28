"""Decision questions answered by an ordinary LLM (design/current.md §6).

One schema-constrained chat call answers every question with `answer`, `confidence` (0-1) and `rationale`.
Answers are normalized per design/current.md §6. For yes/no questions the probability of "yes" comes from the
token logprobs when the model returns them (`calibrated=True`); otherwise from the stated confidence
(`calibrated=False`). A question whose answer is missing or invalid gets its own `error`.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..structured import parse_json
from ..text import TextClient, TextResult
from .questions import answer, failed, normalized_score

SYSTEM = (
    "You judge the state given by the user by answering every question below. For each question reply "
    "with `answer`, `confidence` (a number from 0 to 1: how sure you are) and a one-sentence `rationale`."
)


def _describe(name: str, q: Mapping[str, Any]) -> str:
    if q["type"] == "yes_no":
        expected = 'answer "yes" or "no"'
    elif q["type"] == "choice":
        expected = "answer exactly one of " + ", ".join(json.dumps(o) for o in q["options"])
    else:
        low, high = q["scale"]
        expected = f"answer a number from {low} to {high}"
        anchors: Mapping[str, str] = q.get("anchors") or {}
        if anchors:
            expected += "; " + ", ".join(f"{k} = {v}" for k, v in anchors.items())
    return f'- "{name}" ({expected}): {q["instructions"]}'


def _answer_schema(q: Mapping[str, Any]) -> dict[str, Any]:
    if q["type"] == "yes_no":
        return {"type": "string", "enum": ["yes", "no"]}
    if q["type"] == "choice":
        return {"type": "string", "enum": list(q["options"])}
    low, high = q["scale"]
    return {"type": "number", "minimum": low, "maximum": high}


def decision_schema(questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """JSON Schema for one reply answering all `questions`."""
    item = {"type": "object", "required": ["answer", "confidence", "rationale"]}
    properties = {
        name: {
            **item,
            "properties": {
                "answer": _answer_schema(q),
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
            },
        }
        for name, q in questions.items()
    }
    return {"type": "object", "properties": properties, "required": list(questions)}


def decision_messages(
    state: str | Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]], images: Sequence[str]
) -> list[dict[str, Any]]:
    """System message with the questions; user message with the state (and images)."""
    listing = "\n".join(_describe(name, q) for name, q in questions.items())
    state_text = state if isinstance(state, str) else json.dumps(state, indent=2, ensure_ascii=False)
    user: Any = state_text
    if images:
        user = [{"type": "text", "text": state_text}, *({"type": "image", "path": p} for p in images)]
    return [
        {"role": "system", "content": f"{SYSTEM}\n\nQuestions:\n{listing}"},
        {"role": "user", "content": user},
    ]


def emulate(
    llm: TextClient,
    state: str | Mapping[str, Any],
    questions: Mapping[str, Mapping[str, Any]],
    images: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    """Answer `questions` about `state` with one call to `llm`."""
    use_logprobs = llm.config.capabilities.logprobs is True
    r = llm.complete(
        decision_messages(state, questions, images),
        schema=decision_schema(questions),
        temperature=0.0,
        logprobs=use_logprobs,
    )
    replies = r.parsed if r.parsed is not None else _salvage(r)
    if not isinstance(replies, dict):
        error = r.error or "the model did not return a JSON object"
        return {name: failed(q["type"], error) for name, q in questions.items()}
    names = [n for n, q in questions.items() if q["type"] == "yes_no"]
    p_yes = yes_probabilities(r.text, r.logprobs, names) if r.logprobs else {}
    by_name: dict[str, Any] = replies  # pyright: ignore[reportUnknownVariableType]
    return {name: normalize(q, by_name.get(name), p_yes.get(name)) for name, q in questions.items()}


def _salvage(r: TextResult) -> Any:
    """The reply as JSON even though it failed schema validation, so valid answers are kept."""
    try:
        return parse_json(r.text)[0]
    except ValueError:
        return None


def normalize(q: Mapping[str, Any], item: Any, p_yes: float | None = None) -> dict[str, Any]:
    """One model reply item -> an `Answer` (design/current.md §6), or an answer with `error` set."""
    if not isinstance(item, dict):
        return failed(q["type"], "the model gave no answer for this question")
    fields: dict[str, Any] = item  # pyright: ignore[reportUnknownVariableType]
    try:
        return NORMALIZE[q["type"]](q, fields, p_yes)
    except (KeyError, TypeError, ValueError) as exc:
        return failed(q["type"], f"invalid answer {json.dumps(fields.get('answer'))}: {exc}")


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{what} must be a number")
    return float(value)


def _confidence(item: Mapping[str, Any]) -> float:
    c = _number(item.get("confidence"), "confidence")
    if not 0.0 <= c <= 1.0:
        raise ValueError(f"confidence {c} is not between 0 and 1")
    return c


def _rationale(item: Mapping[str, Any]) -> str | None:
    value = item.get("rationale")
    return str(value) if value is not None else None


def _yes_no(q: Mapping[str, Any], item: Mapping[str, Any], p_yes: float | None) -> dict[str, Any]:
    said = str(item["answer"]).strip().lower()
    if said not in ("yes", "no"):
        raise ValueError("expected yes or no")
    c = _confidence(item)
    p = p_yes if p_yes is not None else (c if said == "yes" else 1.0 - c)
    return answer(
        "yes_no",
        value=p,
        probabilities={"yes": p, "no": 1.0 - p},
        confidence=c,
        calibrated=p_yes is not None,
        rationale=_rationale(item),
    )


def _choice(q: Mapping[str, Any], item: Mapping[str, Any], _: float | None) -> dict[str, Any]:
    options: list[str] = list(q["options"])
    chosen = item["answer"]
    if chosen not in options:
        raise ValueError(f"expected one of {options}")
    c = _confidence(item)
    rest = (1.0 - c) / (len(options) - 1)  # stated confidence spread evenly over the other options
    return answer(
        "choice",
        value=c,
        choice=chosen,
        probabilities={o: c if o == chosen else rest for o in options},
        confidence=c,
        rationale=_rationale(item),
    )


def _score(q: Mapping[str, Any], item: Mapping[str, Any], _: float | None) -> dict[str, Any]:
    raw = _number(item["answer"], "a score")
    return answer(
        "score",
        value=normalized_score(raw, q["scale"]),
        raw=raw,
        confidence=_confidence(item),
        rationale=_rationale(item),
    )


NORMALIZE: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any], float | None], dict[str, Any]]] = {
    "yes_no": _yes_no,
    "choice": _choice,
    "score": _score,
}


def yes_probabilities(text: str, logprobs: list[dict[str, Any]], names: Sequence[str]) -> dict[str, float]:
    """P("yes") per yes/no question from the logprobs of the token where its answer starts.

    Only used when the tokens spell out exactly `text`; questions whose token cannot be found are left out.
    """
    if "".join(t["token"] for t in logprobs) != text:
        return {}
    out: dict[str, float] = {}
    for name in names:
        match = re.search(rf'"{re.escape(name)}"\s*:\s*\{{[^{{}}]*?"answer"\s*:\s*"', text)
        if match is None:
            continue
        token = _token_at(logprobs, match.end())
        p = _yes_share(token["top"]) if token else None
        if p is not None:
            out[name] = p
    return out


def _token_at(logprobs: list[dict[str, Any]], pos: int) -> dict[str, Any] | None:
    start = 0
    for token in logprobs:
        end = start + len(token["token"])
        if start <= pos < end:
            return token
        start = end
    return None


def _yes_share(top: Mapping[str, float]) -> float | None:
    """Probability mass of yes-tokens over yes + no tokens among the top alternatives."""
    mass = {"y": 0.0, "n": 0.0}
    for token, logprob in top.items():
        word = token.strip(' "').lower()
        if word in ("yes", "y", "no", "n"):
            mass[word[0]] += math.exp(logprob)
    total = mass["y"] + mass["n"]
    return mass["y"] / total if total > 0 else None
