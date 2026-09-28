"""LiteLLM provider (extra `litellm`): any hosted chat or embedding model LiteLLM knows.

Requests and replies use the OpenAI shape (shared with `openai_compat`). Capability hints for ad-hoc ids
(`litellm:anthropic/claude-sonnet-5`) come from LiteLLM's model map. LiteLLM is imported on first use.
"""

from __future__ import annotations

import importlib
from typing import Any

from .._http import api_key
from ..errors import ConfigError, ModelTimeout, ProviderError
from ..registry import Capabilities, ModelConfig, Price
from .common import ChatReply, ChatRequest
from .openai_compat import embedding_rows, parse_reply, request_body

RETRIES = 2  # LiteLLM retries transient errors itself; not recorded as span events (decisions D-015)


def _litellm() -> Any:
    try:
        return importlib.import_module("litellm")
    except ImportError:
        raise ConfigError(
            "LiteLLM models need the 'litellm' extra: pip install 'hone-models[litellm]'"
        ) from None


def _call(fn: str, cfg: ModelConfig, timeout_s: float, **payload: Any) -> Any:
    """`litellm.<fn>(...)` with LiteLLM errors mapped to ours; returns the response as a dict."""
    litellm = _litellm()
    extra = {"api_key": api_key(cfg), "api_base": cfg.base_url}
    try:
        response = getattr(litellm, fn)(
            **payload, **{k: v for k, v in extra.items() if v}, timeout=timeout_s, num_retries=RETRIES
        )
    except litellm.exceptions.Timeout as exc:
        raise ModelTimeout(f"LiteLLM model {cfg.name} did not answer within {timeout_s:.0f}s") from exc
    except Exception as exc:  # LiteLLM raises many types; all are provider failures here
        status = getattr(exc, "status_code", None)
        raise ProviderError(f"LiteLLM {fn} for {cfg.name} failed: {exc}", status=status) from exc
    return response.model_dump()


def chat(cfg: ModelConfig, req: ChatRequest) -> ChatReply:
    """One `litellm.completion` call (non-streaming)."""
    data = _call("completion", cfg, req.timeout_s, **request_body(cfg, req))
    return parse_reply(cfg, data, f"LiteLLM {cfg.name}")


def embed(cfg: ModelConfig, texts: list[str], timeout_s: float) -> list[list[float]]:
    """One `litellm.embedding` call; vectors in input order."""
    data = _call("embedding", cfg, timeout_s, model=cfg.name, input=texts)
    return embedding_rows(data, f"LiteLLM {cfg.name}")


def probe(cfg: ModelConfig) -> Capabilities:
    """Capability hints from LiteLLM's model map (all unknown when the model is not in it)."""
    litellm = _litellm()
    try:
        info: dict[str, Any] = litellm.get_model_info(cfg.name)
    except Exception:  # not in the map: capabilities stay unknown
        return Capabilities()
    price = None
    if info.get("input_cost_per_token") is not None and info.get("output_cost_per_token") is not None:
        price = Price(
            input_per_mtok=info["input_cost_per_token"] * 1e6,
            output_per_mtok=info["output_cost_per_token"] * 1e6,
        )
    return Capabilities(
        vision=info.get("supports_vision"),
        thinking=info.get("supports_reasoning"),
        json_schema=info.get("supports_response_schema"),
        max_input_tokens=info.get("max_input_tokens"),
        max_output_tokens=info.get("max_output_tokens"),
        price=price,
    )
