"""What the text client sends to a provider and gets back. Providers are plain modules with
`chat(cfg, request) -> ChatReply` (and `embed(cfg, texts, timeout)` where supported)."""

from __future__ import annotations

import gc
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..errors import ConfigError, ProviderError

Style = tuple["str | None", "float | None"]  # (emotion, intensity) of a speech call


class SpeechEngine(Protocol):
    """A loaded speech model: call it for many texts, then `close()` it to free its memory."""

    def __call__(self, chunks: list[str], voice: str, speed: float, style: Style) -> tuple[int, list[bytes]]:
        """16-bit mono PCM for each chunk and its sample rate."""
        ...

    def close(self) -> None:
        """Free the model (and its GPU memory)."""
        ...


@contextmanager
def library_errors(what: str) -> Generator[None]:
    """Turn the plain exceptions of a model library (download, CUDA, audio) into a `ProviderError`."""
    try:
        yield
    except (ConfigError, ProviderError):
        raise
    except Exception as exc:
        raise ProviderError(f"{what} failed: {type(exc).__name__}: {exc}") from exc


def free_torch(torch: Any) -> None:
    """Collect garbage and give cached CUDA memory back, so nothing lingers on a shared GPU."""
    gc.collect()
    if torch.cuda.is_available():
        clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        if clear_workspaces is not None:
            clear_workspaces()  # cuBLAS keeps about 10 MB per process otherwise
        torch.cuda.empty_cache()


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """A provider-neutral chat request.

    `messages` use the port format; image parts are already `{"type": "image", "data_b64", "mime"}`.
    `schema` is set only when the model supports constrained output. `params` hold sampling options
    (`temperature`, `top_p`, `seed`, `stop`, ...) and always `max_tokens`.
    """

    messages: list[dict[str, Any]]
    params: dict[str, Any]
    timeout_s: float
    schema: dict[str, Any] | None = None
    num_ctx: int | None = None
    think: bool | None = None
    logprobs: bool = False


@dataclass(frozen=True, slots=True)
class ChatReply:
    """A provider-neutral chat reply. `logprobs`: `[{"token", "logprob", "top": {token: logprob}}]`."""

    text: str
    finish_reason: str | None
    model: str
    usage: dict[str, int] = field(default_factory=dict[str, int])
    thinking: str = ""
    logprobs: list[dict[str, Any]] | None = None
    digest: str | None = None


def text_of(content: Any) -> str:
    """The text of a message `content` (a string, or a list of parts)."""
    if isinstance(content, str):
        return content
    parts: list[Any] = list(content or [])
    return "\n".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def images_of(content: Any) -> list[dict[str, Any]]:
    """The image parts of a message `content`."""
    if isinstance(content, str):
        return []
    return [p for p in content or [] if isinstance(p, dict) and p.get("type") == "image"]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
