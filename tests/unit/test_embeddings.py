import json

import pytest
import respx

import hone_models as mk
from hone_models.errors import ConfigError, ProviderError
from select_contracts import check_embedder

URL = "http://127.0.0.1:11434"


def test_ollama_embeddings_normalized_and_recorded() -> None:
    sink = mk.records.MemorySink()
    with respx.mock(base_url=URL) as mock:
        route = mock.post("/api/embed").respond(json={"embeddings": [[3.0, 4.0], [0.0, 2.0]]})
        emb = mk.embedder("ollama:tiny-embed", sink=sink)
        vecs = emb.embed(["a song about rain", "a song about the sea"])
    assert vecs == [[0.6, 0.8], [0.0, 1.0]]
    assert emb.dimensions == 2
    assert json.loads(route.calls.last.request.content) == {
        "model": "tiny-embed",
        "input": ["a song about rain", "a song about the sea"],
    }
    span = sink.spans[0]
    assert span["name"] == "hone.models.embed"
    assert span["attributes"]["gen_ai.operation.name"] == "embeddings"
    assert span["attributes"]["gen_ai.embeddings.dimension.count"] == 2
    assert span["attributes"]["hone.models.embed.count"] == 2


def test_dimensions_probe_and_empty_input() -> None:
    with respx.mock(base_url=URL) as mock:
        route = mock.post("/api/embed").respond(json={"embeddings": [[1.0, 0.0, 0.0]]})
        emb = mk.embedder("ollama:tiny-embed", sink=mk.records.NullSink())
        assert emb.dimensions == 3
        assert emb.dimensions == 3
        assert emb.embed([]) == []
    assert route.call_count == 1


def test_embedding_errors() -> None:
    sink = mk.records.MemorySink()
    emb = mk.embedder("nomic-embed-text", sink=sink)  # 768 dimensions registered
    with respx.mock(base_url=URL) as mock:
        route = mock.post("/api/embed")
        route.respond(json={"embeddings": [[1.0, 0.0]]})
        with pytest.raises(ProviderError, match="expected 768"):
            emb.embed(["a"])
        route.respond(json={"embeddings": [[1.0] * 768]})
        with pytest.raises(ProviderError, match="1 vectors for 2 texts"):
            emb.embed(["a", "b"])
        route.respond(json={"embeddings": [[0.0] * 768]})
        with pytest.raises(ProviderError, match="all-zero"):
            emb.embed(["a"])
    assert all(s["status"]["code"] == "error" for s in sink.spans)
    with pytest.raises(ConfigError, match="not an embedding model"):
        mk.embedder("gemma4-12b")


def test_openai_compatible_embeddings_contract(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0000000000000000000")
    reply = {"data": [{"index": 1, "embedding": [0.0, 5.0]}, {"index": 0, "embedding": [2.0, 0.0]}]}
    with respx.mock(base_url="https://api.openai.com/v1") as mock:
        mock.post("/embeddings").respond(json=reply)
        emb = mk.embedder("openai:text-embedding-3-small", sink=mk.records.NullSink())
        assert emb.embed(["x", "y"]) == [[1.0, 0.0], [0.0, 1.0]]  # sorted by index
        check_embedder(emb)


@pytest.mark.parametrize(
    ("model", "path", "body"),
    [
        ("ollama:tiny-embed", "http://127.0.0.1:11434/api/embed", {"error": "model not found"}),
        ("openai:text-embedding-3-small", "https://api.openai.com/v1/embeddings", {"object": "list"}),
        ("openai:text-embedding-3-small", "https://api.openai.com/v1/embeddings", {"data": [{"index": 0}]}),
    ],
)
def test_malformed_embed_response_is_a_provider_error(model, path, body, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sink = mk.records.MemorySink()
    with respx.mock() as mock:
        mock.post(path).respond(json=body)
        with pytest.raises(ProviderError, match="returned no embeddings"):
            mk.embedder(model, sink=sink).embed(["a"])
    assert sink.spans[0]["status"]["code"] == "error"


def test_embedder_rejects_models_without_embeddings() -> None:
    with pytest.raises(ConfigError, match="decision model, not an embedding model"):
        mk.embedder("jev")
    with pytest.raises(ConfigError, match=r"\['litellm', 'ollama', 'openai_compatible'\]"):
        mk.embedder("jev:some-model", sink=mk.records.NullSink()).embed(["a"])
