"""Jev (TypeSafe AI), a native decision model: `state` + `questions` in, probabilities and scores out.

**Unverified:** the API shape below is reconstructed from public articles (Sept 2026), not from an API
reference. It is kept to two small pure functions, `to_request` and `from_response`, tested against
recorded fixtures (`tests/fixtures/http/jev_*.json`), so fixing it later touches only this module.

    POST {base_url}/decide     Authorization: Bearer $TYPESAFE_API_KEY
    {"model": "jev", "state": "...", "questions": [{"id", "type", "text", "options"?, "scale"?, "anchors"?}],
     "images": [{"data": "<base64>", "mime": "image/png"}]}
    -> {"model": "jev-1", "answers": [{"id", "p_yes"} | {"id", "probabilities"} | {"id", "score"}
                                      | {"id", "error"}],
        "usage": {"input_tokens": 57}}
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .._http import bearer_headers, request_json
from ..decision.questions import answer, failed, normalized_score, probability
from ..errors import ProviderError
from ..registry import ModelConfig

DEFAULT_URL = "https://api.typesafe.ai/v1"


def to_request(
    cfg: ModelConfig,
    state: str | Mapping[str, Any],
    questions: Mapping[str, Mapping[str, Any]],
    images: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The request body. `images` are inline parts `{"type": "image", "data_b64", "mime"}`."""
    items: list[dict[str, Any]] = []
    for name, q in questions.items():
        item = {"id": name, "type": q["type"], "text": q["instructions"]}
        item.update({k: q[k] for k in ("options", "scale", "anchors") if k in q})
        items.append(item)
    body: dict[str, Any] = {"model": cfg.name, "state": state, "questions": items}
    if images:
        body["images"] = [{"data": p["data_b64"], "mime": p["mime"]} for p in images]
    return body


def from_response(
    data: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]], *, calibrated: bool
) -> dict[str, dict[str, Any]]:
    """Answers (design/current.md §6) from the response body; unanswered questions get an `error`."""
    items: list[Any] = data.get("answers") or []
    got: dict[str, dict[str, Any]] = {
        i["id"]: i
        for i in items
        if isinstance(i, dict) and "id" in i  # pyright: ignore[reportUnknownArgumentType]
    }
    out: dict[str, dict[str, Any]] = {}
    for name, q in questions.items():
        item = got.get(name)
        if item is None or item.get("error"):
            reason = item.get("error") if item else None
            out[name] = failed(q["type"], f"jev did not answer: {reason or 'missing from the response'}")
            continue
        try:
            out[name] = _answer(q, item) | {"calibrated": calibrated}
        except (KeyError, TypeError, ValueError) as exc:
            out[name] = failed(q["type"], f"unexpected jev answer {dict(item)}: {exc}")
    return out


def _answer(q: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    if q["type"] == "yes_no":
        p = probability(item["p_yes"])
        return answer("yes_no", value=p, probabilities={"yes": p, "no": 1.0 - p})
    if q["type"] == "choice":
        probs = {str(k): probability(v) for k, v in item["probabilities"].items()}
        if not set(probs) <= set(q["options"]):
            raise ValueError(f"{sorted(set(probs) - set(q['options']))} not one of the options")
        if sum(probs.values()) > 1.0 + 1e-6:
            raise ValueError("choice probabilities sum above 1")
        chosen = max(probs, key=lambda o: probs[o])
        return answer("choice", value=probs[chosen], choice=chosen, probabilities=probs)
    raw = float(item["score"])
    return answer("score", value=normalized_score(raw, q["scale"]), raw=raw)


def decide(
    cfg: ModelConfig,
    state: str | Mapping[str, Any],
    questions: Mapping[str, Mapping[str, Any]],
    images: Sequence[Mapping[str, Any]],
    timeout_s: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, int], str]:
    """One `/decide` call: (answers, usage, model name reported by the server)."""
    url = (cfg.base_url or DEFAULT_URL).rstrip("/") + "/decide"
    body = to_request(cfg, state, questions, images)
    data = request_json("POST", url, payload=body, headers=bearer_headers(cfg), timeout=timeout_s)
    if not isinstance(data, dict):
        raise ProviderError(f"{url} returned {type(data).__name__}, expected a JSON object")
    reply: dict[str, Any] = data  # pyright: ignore[reportUnknownVariableType]
    answers = from_response(reply, questions, calibrated=cfg.capabilities.calibrated is not False)
    counts: dict[str, Any] = reply.get("usage") or {}
    usage = {k: int(v) for k, v in counts.items() if k in ("input_tokens", "output_tokens")}
    return answers, usage, str(reply.get("model") or cfg.name)
