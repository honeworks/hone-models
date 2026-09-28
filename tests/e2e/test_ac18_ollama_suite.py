"""AC-18 with the packaged FakeOllama (the real-model version is tests/gpu/test_ac18_real_ollama.py):
chat, JSON schema, decision emulation, an image, embeddings, think handling, contract checkers, unload."""

import pytest
from pydantic import BaseModel

import hone_models as mk
from hone_models.testing import FakeOllama
from select_contracts import check_decision_client, check_embedder, check_text_client

pytestmark = pytest.mark.e2e


class Verdict(BaseModel):
    score: int
    rationale: str


def test_ac18_full_ollama_cycle_against_a_fake_server(isolated) -> None:
    (isolated / "frame.png").write_bytes(b"\x89PNG fake")
    sink = mk.records.MemorySink()
    with FakeOllama() as server:
        llm = mk.text("gemma4-12b", sink=sink)
        assert llm.complete([{"role": "user", "content": "hello"}]).text == "echo: hello"
        r = llm.complete([{"role": "user", "content": "Rate it."}], schema=Verdict)
        assert isinstance(r.parsed, Verdict)
        assert r.structured_path == "constrained"
        check_text_client(llm)

        dm = mk.decision("gemma4-12b", sink=sink)
        check_decision_client(dm)
        assert dm.decide("x", {"ok": mk.YesNo("Ok?")})["ok"]["error"] is None

        image = [
            {"type": "text", "text": "Describe."},
            {"type": "image", "path": str(isolated / "frame.png")},
        ]
        assert mk.text("qwen2.5vl-7b", sink=sink).complete([{"role": "user", "content": image}]).error is None

        check_embedder(mk.embedder("ollama:nomic-embed-text", sink=sink))
        mk.unload("gemma4-12b")

    chats = [q["body"] for q in server.requests if q["path"] == "/api/chat"]
    assert chats[0]["think"] is False  # gemma4-12b declares thinking: reasoning off unless asked
    assert any("images" in m for c in chats for m in c["messages"])
    unload = [q["body"] for q in server.requests if q["path"] == "/api/generate"]
    assert unload == [{"model": "gemma4-12b:latest", "keep_alive": 0}]
    assert all(s["status"]["code"] == "ok" for s in sink.spans)
