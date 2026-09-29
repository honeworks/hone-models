"""What the clients send to a provider and get back. Providers are plain modules with
`chat(cfg, request) -> ChatReply` (and `embed(cfg, texts, timeout)` where supported); speech providers
give a `SpeechEngine`, generation providers a `MediaProvider`."""

from __future__ import annotations

import gc
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..errors import ConfigError, ProviderError
from ..registry import ModelConfig

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


# The shared vocabulary of named inputs (change 0015 §1); a model may take more (its entry lists them).
COMMON_INPUTS = (
    "negative",
    "size",
    "references",
    "image",
    "source",
    "strength",
    "lyrics",
    "duration_s",
    "n",
    "steps",
)
FILE_INPUTS = ("references", "image", "source")  # paths, also when given as strings
ERROR_KINDS = ("out_of_memory", "refused", "invalid_input", "no_output", "failed")
_ERROR_WORDS = {
    "out_of_memory": ("out of memory", "outofmemory", "cuda oom", "not enough memory"),
    "refused": ("moderation", "content policy", "safety system", "refus", "blocked"),
    "invalid_input": ("invalid", "valueerror", "must be", "unsupported", "not supported"),
}


@dataclass(slots=True)
class MediaJob:
    """One generation call, as the media client hands it to a provider.

    `inputs` are checked (only names the provider accepts, defaults applied, files exist, sizes and
    durations the entry declares); `seed` is the one to use; the provider writes its files next to `out`
    (`_media_files.output_paths`). `attrs` are the call's span attributes: a provider adds its
    `hone.models.media.*` keys there (job id, workflow hash, loaded / freed / server_started, ...), so they
    are recorded even when it raises. It enters `lease()` around the work that uses the GPU (a no-op
    for hosted models and inside a session). `session` is the state its `session()` yielded, `None`
    outside a session."""

    config: ModelConfig
    prompt: str
    inputs: dict[str, Any]
    seed: int
    out: Path
    timeout_s: float
    attrs: dict[str, Any]
    lease: Callable[[], AbstractContextManager[Any]]
    session: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class MediaOutcome:
    """What a job produced: the files it wrote, the provider's job id, and for a job that ran without
    usable output the provider's message and its kind (one of `ERROR_KINDS`)."""

    files: list[Path]
    job_id: str | None = None
    error: str | None = None
    error_kind: str | None = None


@contextmanager
def no_session(cfg: ModelConfig) -> Generator[dict[str, Any]]:
    """The session of a provider that keeps nothing between calls."""
    yield {}


@dataclass(frozen=True, slots=True)
class MediaProvider:
    """A generation provider: which named inputs an entry takes (besides `prompt` and `seed`), how to
    run one job (a job that ran without usable output is a `MediaOutcome` with `error`; one that could
    not be submitted, run or fetched raises), and what to hold for a `client.session()` block."""

    accepted: Callable[[ModelConfig], set[str]]
    run: Callable[[MediaJob], MediaOutcome]
    session: Callable[[ModelConfig], AbstractContextManager[dict[str, Any]]] = no_session


def listed_inputs(cfg: ModelConfig) -> set[str]:
    """The shared vocabulary plus the extra names a non-ComfyUI entry lists in `inputs`."""
    extra = cfg.inputs if isinstance(cfg.inputs, list) else []
    return {*COMMON_INPUTS, *extra}


def error_kind(message: str) -> str:
    """The kind of a provider's error message: `out_of_memory`, `refused`, `invalid_input` or `failed`."""
    lowered = message.lower()
    for kind, words in _ERROR_WORDS.items():
        if any(word in lowered for word in words):
            return kind
    return "failed"
