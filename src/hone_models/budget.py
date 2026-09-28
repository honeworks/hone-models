"""Sizing a call before sending it: prompt tokens, output budget, context window, timeout.

Nothing is truncated silently: if prompt + requested output do not fit, `ContextOverflow` is raised
before any HTTP call, with the numbers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ._images import image_size
from .errors import ConfigError, ContextOverflow
from .providers.common import images_of, text_of
from .registry import ModelConfig

CHARS_PER_TOKEN = 3.5
DEFAULT_MAX_TOKENS = 2048
CTX_STEP = 2048  # round num_ctx up so Ollama doesn't reload the model for every small change
MIN_TIMEOUT_S = 120.0
DEFAULT_IMAGE_TOKENS = 1024  # per image, when the model declares neither image_tokens nor image_patch_px
ASSUMED_LOCAL_TOK_S = 10.0  # a slow local model (partly offloaded on an 8 GB card), until speed is measured


def estimate_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Rough prompt text size: characters / 3.5 (images are counted by `image_tokens`)."""
    chars = sum(len(text_of(m.get("content"))) for m in messages)
    return math.ceil(chars / CHARS_PER_TOKEN)


def image_tokens(cfg: ModelConfig, messages: Sequence[Mapping[str, Any]]) -> int:
    """What the images in `messages` cost (change 0013): the model's flat `image_tokens` per image, else
    one token per `image_patch_px` square of the image (read from its header), else 1024 per image."""
    caps = cfg.capabilities
    total = 0
    for part in (p for m in messages for p in images_of(m.get("content"))):
        patch = caps.image_patch_px
        size = image_size(part) if caps.image_tokens is None and patch else None
        if caps.image_tokens is not None:
            total += caps.image_tokens
        elif size and patch:
            total += math.ceil(size[0] / patch) * math.ceil(size[1] / patch)
        else:
            total += DEFAULT_IMAGE_TOKENS
    return total


def output_budget(cfg: ModelConfig, params: Mapping[str, Any]) -> int:
    """Requested output tokens: `max_tokens` param, else the model's max output, else 2048."""
    wanted = params.get("max_tokens")
    if wanted is None:
        return cfg.capabilities.max_output_tokens or DEFAULT_MAX_TOKENS
    if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 1:
        raise ConfigError(f"max_tokens must be a positive int, got {wanted!r}")
    return wanted


def min_context(params: Mapping[str, Any]) -> int | None:
    """The `min_num_ctx` param (or registry default): a floor for the context size sent."""
    value = params.get("min_num_ctx")
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
        raise ConfigError(f"min_num_ctx must be a positive int, got {value!r}")
    return value


def plan_context(
    cfg: ModelConfig, messages: Sequence[Mapping[str, Any]], max_tokens: int, min_ctx: int | None = None
) -> tuple[int, int, int]:
    """Return (estimated prompt tokens, num_ctx, the images' share of the prompt); raise
    `ContextOverflow` if it cannot fit. `min_ctx` (param or registry default `min_num_ctx`) is a floor
    for num_ctx, for models whose image cost is unknown or to keep one context size."""
    images = image_tokens(cfg, messages)
    estimated = estimate_tokens(messages) + images
    needed = estimated + max_tokens
    limit = cfg.capabilities.max_input_tokens
    if limit is not None and needed > limit:
        shown = f"~{estimated} tokens, {images} of them images" if images else f"~{estimated} tokens"
        raise ContextOverflow(
            f"prompt ({shown}) + requested output ({max_tokens}) = {needed} tokens exceeds "
            f"{cfg.id}'s context of {limit} tokens; shorten the prompt, lower max_tokens, or choose a "
            f"model with require={{'min_context': {needed}}}"
        )
    num_ctx = max(math.ceil(needed * 1.1 / CTX_STEP) * CTX_STEP, min_ctx or 0)
    return estimated, min(num_ctx, limit) if limit else num_ctx, images


def timeout_for(cfg: ModelConfig, max_tokens: int) -> float:
    """Request timeout: at least 120 s, `2 * max_tokens / speed` for long answers, capped by `max_timeout_s`.
    The speed is the measured `speed_tok_s`; a local model without one is assumed slow (10 tokens/s), so
    a long answer is not cut off on a fresh machine (change 0010)."""
    speed = cfg.capabilities.speed_tok_s or (ASSUMED_LOCAL_TOK_S if cfg.local else None)
    expected = 2 * max_tokens / speed if speed else 0.0
    return min(cfg.max_timeout_s, max(MIN_TIMEOUT_S, expected))
