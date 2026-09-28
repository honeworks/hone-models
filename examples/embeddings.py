"""Embeddings: unit-length vectors in input order, and a cosine-similarity ranking.

What: embed a few texts, rank them by similarity to a query, and see the checks on vector size.

How: `emb = mk.embedder(model_id)` (a registered embedding model such as "nomic-embed-text", or an
    ad-hoc id such as "ollama:all-minilm"); `emb.embed(texts)` returns one L2-normalized vector per text,
    in input order (`[]` for no texts). Because the vectors have length 1, cosine similarity is a plain
    dot product. `emb.dimensions` is the registry's `capabilities.dimensions`, else measured once.

Why: dedup, retrieval and diversity checks (hone-select) need comparable vectors from any provider
    (Ollama, OpenAI-compatible, LiteLLM). Vectors of the wrong length raise `ProviderError` instead of
    silently mixing sizes: if a registered `dimensions` is wrong, fix the registry. Pitfall: never
    compare vectors from two different embedding models.

Runs offline: `FakeOllama` returns stable hash vectors, so the ranking is arbitrary but repeatable.
"""

import hone_models as mk
from hone_models.testing import FakeOllama


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two unit vectors: their dot product."""
    return sum(x * y for x, y in zip(a, b, strict=True))


server = FakeOllama().start()  # offline stand-in for Ollama (sets OLLAMA_HOST)
sink = mk.records.MemorySink()

emb = mk.embedder("ollama:all-minilm", sink=sink)  # ad-hoc: the size is measured, not declared
query = "a song about rain"
texts = ["a ballad about drizzle", "corporate tax law", "a storm at sea"]

query_vec = emb.embed([query])[0]
vectors = emb.embed(texts)
ranked = sorted(
    ((cosine(query_vec, vec), text) for text, vec in zip(texts, vectors, strict=True)), reverse=True
)
print("dimensions:", emb.dimensions)
for score, text in ranked:
    print(f"  {score:.3f}  {text}")

assert emb.dimensions == 8  # measured on the first call (FakeOllama's size)
assert emb.embed([texts[1]])[0] == vectors[1]  # input order: the same text gives the same vector
assert all(abs(cosine(v, v) - 1.0) < 1e-9 for v in vectors)  # unit length
assert emb.embed([]) == []  # nothing to embed: no call
print("recorded:", sink.spans[-1]["name"], sink.spans[-1]["attributes"]["hone.models.embed.count"], "texts")

# A registered model declares its size; a server returning another size is an error, not a silent mix.
try:
    mk.embedder("nomic-embed-text", sink=sink).embed(["x"])  # declares 768; the fake returns 8
except mk.errors.ProviderError as exc:
    print("refused:", exc)
else:
    raise AssertionError("a vector size that disagrees with the registry must raise")

server.stop()
