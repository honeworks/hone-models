"""Embeddings: `mk.embedder(...)` implements the embedder shape (design/current.md §10).

emb = mk.embedder("nomic-embed-text")
vecs = emb.embed(["a song about rain", "a song about the sea"])   # L2-normalized, input order
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ._http import register_key
from ._tracing import start_span
from .budget import timeout_for
from .errors import ConfigError, ProviderError
from .ports import RecordSink
from .providers import EMBED, lookup, model_attributes
from .records import default_sink
from .registry import ModelConfig, Registry, load


class Embedder:
    """Embeds texts with one model; records a `hone.models.embed` span per call."""

    def __init__(self, config: ModelConfig, sink: RecordSink) -> None:
        self.config = config
        self.sink = sink
        self._dimensions = config.capabilities.dimensions

    @property
    def model_id(self) -> str:
        return self.config.id

    @property
    def dimensions(self) -> int:
        """Vector length: from the registry, else measured once with a short probe text."""
        if self._dimensions is None:
            self._dimensions = len(self._call(["dimensions"])[0])
        return self._dimensions

    def embed(self, texts: Sequence[str], *, trace: Mapping[str, str] | None = None) -> list[list[float]]:
        """L2-normalized vectors in input order; `[]` for no texts."""
        texts = list(texts)
        if not texts:
            return []
        attrs = {**model_attributes(self.config, "embeddings"), "hone.models.embed.count": len(texts)}
        with start_span("hone.models.embed", self.sink, attrs, trace=trace) as span:
            vectors = [unit(v) for v in self._call(texts)]
            if len(vectors) != len(texts):
                raise ProviderError(f"{self.model_id} returned {len(vectors)} vectors for {len(texts)} texts")
            dims = {len(v) for v in vectors}
            if len(dims) != 1 or (self._dimensions is not None and dims != {self._dimensions}):
                raise ProviderError(
                    f"{self.model_id} returned vectors of length {sorted(dims)}, expected "
                    f"{self._dimensions}; fix capabilities.dimensions in the registry"
                )
            self._dimensions = dims.pop()
            span["attributes"]["gen_ai.embeddings.dimension.count"] = self._dimensions
        return vectors

    def _call(self, texts: list[str]) -> list[list[float]]:
        embed = lookup(EMBED, self.config, "embeddings")
        return embed(self.config, texts, timeout_for(self.config, 0))


def unit(vector: Sequence[float]) -> list[float]:
    """`vector` scaled to length 1; `ProviderError` for an all-zero vector."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise ProviderError("the model returned an all-zero embedding, which cannot be normalized")
    return [x / norm for x in vector]


def embedder(model_id: str, *, registry: Registry | None = None, sink: RecordSink | None = None) -> Embedder:
    """An `Embedder` for a registered embedding model or an ad-hoc id (`ollama:nomic-embed-text`)."""
    reg = registry or load()
    cfg = reg.get(model_id)
    if cfg.kind != "embedding" and cfg.id in reg.models:
        raise ConfigError(f"model {cfg.id!r} is a {cfg.kind} model, not an embedding model")
    register_key(cfg)
    return Embedder(cfg, sink or default_sink())
