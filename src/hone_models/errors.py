"""Typed exceptions raised at the public API.

All subclass `HoneModelsError`, which subclasses `RuntimeError` so consumers that catch
`RuntimeError` at a port boundary (design/current.md §2.2) also catch ours.
"""

from __future__ import annotations


class HoneModelsError(RuntimeError):
    """Base class for every error raised by hone-models."""


class ConfigError(HoneModelsError):
    """A registry file, model id or argument is invalid."""


class CapabilityError(HoneModelsError):
    """The model cannot do what the call needs (vision, context size, ...), detected before calling."""


class ContextOverflow(CapabilityError):  # noqa: N818 - public name (design/current.md §2)
    """Prompt plus requested output does not fit the model's context window."""


class ProviderError(HoneModelsError):
    """The provider failed: unreachable server, HTTP error, malformed response."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ModelTimeout(ProviderError):  # noqa: N818 - public name (design/current.md §2)
    """The provider did not answer within the timeout."""


class ValidationFailed(HoneModelsError):  # noqa: N818 - public name (design/current.md §2)
    """Model output did not match the requested schema (used inside the structured pipeline)."""
