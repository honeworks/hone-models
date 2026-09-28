# Embeddings

```python
import hone_models as mk

emb = mk.embedder("ollama:nomic-embed-text")
vectors = emb.embed(["a song about rain", "a song about the sea"])
print(emb.dimensions, len(vectors))
```

Vectors are L2-normalized (a dot product is the cosine similarity) and in input order; empty input
returns `[]`. `dimensions` comes from the registry (`capabilities.dimensions`) or from a first call, and
a model that returns vectors of another length raises `ProviderError`. Ollama, OpenAI-compatible servers
and LiteLLM are supported.

**Runnable examples:** [embeddings.py](../examples/embeddings.py).
