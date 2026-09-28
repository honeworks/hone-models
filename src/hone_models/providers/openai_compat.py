"""OpenAI-compatible provider: `/chat/completions` and `/embeddings` under the registry `base_url`.

Covers OpenAI, llama.cpp server, vLLM, LM Studio and Ollama's `/v1`. Sends `response_format` with the
JSON Schema when the model supports it and asks for token logprobs when requested.
"""

from __future__ import annotations

from typing import Any

from .._http import bearer_headers, request_json
from ..errors import ConfigError, ProviderError
from ..registry import ModelConfig
from .common import ChatReply, ChatRequest, images_of, text_of

PARAM_NAMES = ("temperature", "top_p", "seed", "stop", "presence_penalty", "frequency_penalty", "max_tokens")
TOP_LOGPROBS = 5
USAGE_FIELDS = (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"))


def _endpoint(cfg: ModelConfig, path: str) -> str:
    if not cfg.base_url:
        raise ConfigError(
            f"model {cfg.id!r} (openai_compatible) needs base_url in the registry, e.g. http://localhost:8080/v1"
        )
    return cfg.base_url.rstrip("/") + path


def _message(msg: dict[str, Any]) -> dict[str, Any]:
    content = msg.get("content")
    images = images_of(content)
    if not images:
        return {"role": msg["role"], "content": text_of(content)}
    parts: list[dict[str, Any]] = [{"type": "text", "text": text_of(content)}]
    parts += [
        {"type": "image_url", "image_url": {"url": f"data:{p['mime']};base64,{p['data_b64']}"}}
        for p in images
    ]
    return {"role": msg["role"], "content": parts}


def chat(cfg: ModelConfig, req: ChatRequest) -> ChatReply:
    """One `/chat/completions` call (non-streaming)."""
    url = _endpoint(cfg, "/chat/completions")
    data = request_json(
        "POST", url, payload=request_body(cfg, req), headers=bearer_headers(cfg), timeout=req.timeout_s
    )
    return parse_reply(cfg, data, url)


def request_body(cfg: ModelConfig, req: ChatRequest) -> dict[str, Any]:
    """The OpenAI chat request body (also used by the LiteLLM provider)."""
    payload: dict[str, Any] = {"model": cfg.name, "messages": [_message(m) for m in req.messages]}
    payload.update({k: req.params[k] for k in PARAM_NAMES if k in req.params})
    if req.schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "output", "schema": req.schema},
        }
    if req.logprobs:
        payload.update(logprobs=True, top_logprobs=TOP_LOGPROBS)
    return payload


def parse_reply(cfg: ModelConfig, data: Any, source: str) -> ChatReply:
    """A `ChatReply` from an OpenAI chat response body; `ProviderError` when it has no choices."""
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"{source} returned no choices: {str(data)[:200]}") from exc
    message: dict[str, Any] = choice.get("message") or {}
    usage: dict[str, Any] = data.get("usage") or {}
    return ChatReply(
        text=message.get("content") or "",
        thinking=message.get("reasoning_content") or message.get("reasoning") or "",
        finish_reason=choice.get("finish_reason"),
        model=data.get("model", cfg.name),
        usage={k: usage[src] for k, src in USAGE_FIELDS if src in usage},
        logprobs=_logprobs(choice),
        digest=data.get("system_fingerprint"),
    )


def _logprobs(choice: dict[str, Any]) -> list[dict[str, Any]] | None:
    """`[{"token", "logprob", "top": {token: logprob}}]` from `choice.logprobs.content`, if present."""
    block: dict[str, Any] = choice.get("logprobs") or {}
    tokens: list[dict[str, Any]] = block.get("content") or []
    out: list[dict[str, Any]] = []
    for t in tokens:
        if "token" not in t or "logprob" not in t:  # some servers send partial entries
            continue
        alternatives: list[dict[str, Any]] = t.get("top_logprobs") or []
        top = {alt["token"]: alt["logprob"] for alt in alternatives if "token" in alt and "logprob" in alt}
        out.append({"token": t["token"], "logprob": t["logprob"], "top": top})
    return out or None


def embed(cfg: ModelConfig, texts: list[str], timeout_s: float) -> list[list[float]]:
    """One `/embeddings` call; vectors in input order."""
    url = _endpoint(cfg, "/embeddings")
    data = request_json(
        "POST",
        url,
        payload={"model": cfg.name, "input": texts},
        headers=bearer_headers(cfg),
        timeout=timeout_s,
    )
    return embedding_rows(data, url)


def embedding_rows(data: Any, source: str) -> list[list[float]]:
    """The vectors of an OpenAI embeddings response, in input order."""
    try:
        rows: list[dict[str, Any]] = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [row["embedding"] for row in rows]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ProviderError(f"{source} returned no embeddings: {str(data)[:200]}") from exc
