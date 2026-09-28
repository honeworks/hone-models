"""Ollama provider: `/api/chat`, `/api/embed`, `/api/show`, `/api/ps`, `/api/generate` (unload).

Notes from OneShotStudio: `format` goes at the top level (inside `options` Ollama ignores it);
thinking models get `think=false` unless reasoning is asked for; `num_ctx` is always set so long
prompts are not silently truncated at the server default.
"""

from __future__ import annotations

import os
from typing import Any

from .._http import bearer_headers, request_json
from ..errors import ProviderError
from ..registry import Capabilities, ModelConfig
from .common import ChatReply, ChatRequest, images_of, text_of

DEFAULT_URL = "http://127.0.0.1:11434"
OPTION_NAMES = ("temperature", "top_p", "top_k", "min_p", "seed", "stop", "repeat_penalty")
USAGE_FIELDS = (("input_tokens", "prompt_eval_count"), ("output_tokens", "eval_count"))
# Models this process loaded, by server URL: the GPU lease may unload them to free memory.
loaded: set[tuple[str, str]] = set()


def base_url(cfg: ModelConfig | None = None) -> str:
    """Server URL: the registry's `base_url`, else `OLLAMA_HOST`, else localhost:11434."""
    url = (cfg.base_url if cfg else None) or os.environ.get("OLLAMA_HOST") or DEFAULT_URL
    url = url if "://" in url else f"http://{url}"
    return url.rstrip("/")


def _message(msg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"role": msg["role"], "content": text_of(msg.get("content"))}
    images = [p["data_b64"] for p in images_of(msg.get("content"))]
    if images:
        out["images"] = images
    return out


def chat(cfg: ModelConfig, req: ChatRequest) -> ChatReply:
    """One `/api/chat` call (non-streaming)."""
    options = {k: req.params[k] for k in OPTION_NAMES if k in req.params}
    options["num_predict"] = req.params["max_tokens"]
    if req.num_ctx:
        options["num_ctx"] = req.num_ctx
    payload: dict[str, Any] = {
        "model": cfg.name,
        "messages": [_message(m) for m in req.messages],
        "stream": False,
        "options": options,
    }
    if req.schema is not None:
        payload["format"] = req.schema
    if req.think is not None:
        payload["think"] = req.think
    url = base_url(cfg)
    data = request_json(
        "POST", f"{url}/api/chat", payload=payload, headers=bearer_headers(cfg), timeout=req.timeout_s
    )
    loaded.add((url, cfg.name))
    message: dict[str, Any] = data.get("message") or {}
    return ChatReply(
        text=message.get("content") or "",
        thinking=message.get("thinking") or "",
        finish_reason=data.get("done_reason"),
        model=data.get("model", cfg.name),
        usage={k: data[src] for k, src in USAGE_FIELDS if src in data},
    )


def embed(cfg: ModelConfig, texts: list[str], timeout_s: float) -> list[list[float]]:
    """One `/api/embed` call."""
    url = base_url(cfg)
    payload = {"model": cfg.name, "input": texts}
    data = request_json(
        "POST", f"{url}/api/embed", payload=payload, headers=bearer_headers(cfg), timeout=timeout_s
    )
    loaded.add((url, cfg.name))
    reply: dict[str, Any] = data if isinstance(data, dict) else {}  # pyright: ignore[reportUnknownVariableType]
    vectors = reply.get("embeddings")
    if not isinstance(vectors, list):
        raise ProviderError(f"{url}/api/embed returned no embeddings: {str(reply)[:200]}")
    return vectors  # pyright: ignore[reportUnknownVariableType]


def probe(cfg: ModelConfig) -> Capabilities:
    """Capabilities reported by `/api/show` (vision, thinking, context length)."""
    data = request_json("POST", f"{base_url(cfg)}/api/show", payload={"model": cfg.name}, timeout=30)
    caps = set(data.get("capabilities") or [])
    info: dict[str, Any] = data.get("model_info") or {}
    context = next((v for k, v in info.items() if k.endswith(".context_length")), None)
    return Capabilities(
        vision="vision" in caps,
        thinking="thinking" in caps,
        json_schema=True,
        max_input_tokens=context,
    )


def unload(url: str, model: str) -> None:
    """Ask the server at `url` to free `model`'s memory now (`keep_alive: 0`)."""
    request_json("POST", f"{url}/api/generate", payload={"model": model, "keep_alive": 0}, timeout=60)
    loaded.discard((url, model))


def running() -> set[tuple[str, str]]:
    """The models the default server holds in memory (`/api/ps`), as `(url, model)` pairs."""
    url = base_url()
    models: list[dict[str, Any]] = request_json("GET", f"{url}/api/ps", timeout=10).get("models") or []
    return {(url, m["name"]) for m in models}
