"""HTTP with retries: transient failures (connect errors, 429, 5xx) retry with jittered backoff,
other 4xx fail at once. Each retry is recorded as a `retry` event on the current span."""

from __future__ import annotations

import os
import random
import time
from typing import Any

import httpx

from ._tracing import add_event
from .errors import ConfigError, ModelTimeout, ProviderError
from .records import add_secret
from .registry import ModelConfig

ATTEMPTS = 3
BACKOFF_S = 0.5
TRANSIENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError)


def request_json(
    method: str, url: str, *, payload: Any = None, headers: dict[str, str] | None = None, timeout: float
) -> Any:
    """Send a JSON request and return the decoded JSON body, retrying transient failures."""
    reason = ""
    for attempt in range(1, ATTEMPTS + 1):
        if attempt > 1:
            add_event("retry", {"attempt": attempt, "reason": reason})
            time.sleep(BACKOFF_S * 2 ** (attempt - 2) * random.uniform(0.5, 1.0))  # noqa: S311 - jitter
        try:
            resp = httpx.request(method, url, json=payload, headers=headers, timeout=timeout)
        except TRANSIENT as exc:
            reason = f"{type(exc).__name__}: {exc}"
            continue
        except httpx.TimeoutException as exc:
            raise ModelTimeout(
                f"{url} did not answer within {timeout:.0f}s; set capabilities.speed_tok_s or max_timeout_s "
                "in the registry for slow models"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"request to {url} failed: {exc}") from exc
        if resp.status_code == 429 or resp.status_code >= 500:
            reason = f"http {resp.status_code}"
            continue
        if resp.status_code >= 400:
            message = f"{url} returned HTTP {resp.status_code}: {resp.text[:500]}"
            raise ProviderError(message, status=resp.status_code)
        try:
            return resp.json()
        except ValueError as exc:
            raise ProviderError(f"{url} returned invalid JSON: {resp.text[:200]!r}") from exc
    raise ProviderError(f"{url} failed after {ATTEMPTS} attempts ({reason}); is the server running?")


def register_key(cfg: ModelConfig) -> None:
    """Mark the model's API key (if set in the environment) as a secret before anything is recorded."""
    if cfg.api_key_env:
        add_secret(os.environ.get(cfg.api_key_env))


def api_key(cfg: ModelConfig) -> str | None:
    """The API key from `cfg.api_key_env` (registered as a secret so it is never recorded)."""
    if not cfg.api_key_env:
        return None
    value = os.environ.get(cfg.api_key_env)
    if not value:  # message kept short: a secret-free hint at what to set
        raise ConfigError(
            f"model {cfg.id!r} needs an API key: set the environment variable {cfg.api_key_env}"
        )
    add_secret(value)
    return value


def bearer_headers(cfg: ModelConfig) -> dict[str, str] | None:
    """`Authorization: Bearer <key>` when the model has an `api_key_env`."""
    key = api_key(cfg)
    return {"Authorization": f"Bearer {key}"} if key else None
