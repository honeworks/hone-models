"""AC-18 [real]: chat, JSON schema, decisions, vision, embeddings and thinking on local Ollama models.

Run through the machine-wide lock: `scripts/gpu-lock.sh uv run pytest -m gpu`. Models come from
HONE_TEST_*_MODEL; tests skip with a reason when Ollama or a model is missing. Models are unloaded after.
"""

import math
import struct
import zlib
from pathlib import Path

import pytest
from pydantic import BaseModel

import hone_models as mk
from select_contracts import check_decision_client, check_embedder, check_text_client

pytestmark = [pytest.mark.gpu, pytest.mark.ollama]


@pytest.fixture(autouse=True)
def ollama_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from conftest import OLLAMA_URL  # noqa: PLC0415 - the shared test setting

    monkeypatch.setenv("OLLAMA_HOST", OLLAMA_URL)


class Verdict(BaseModel):
    score: int
    rationale: str


def red_png(path: Path, size: int = 64) -> Path:
    """A plain red PNG written with the standard library."""
    row = b"\x00" + b"\xff\x00\x00" * size

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    body = chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(row * size)) + chunk(b"IEND", b"")
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + body)
    return path


def test_ac18_text_chat_json_and_decisions(ollama_model) -> None:
    name = ollama_model("HONE_TEST_TEXT_MODEL", "gemma4-12b:latest")
    sink = mk.records.MemorySink()
    llm = mk.text(f"ollama:{name}", sink=sink)

    r = llm.complete([{"role": "user", "content": "Reply with the single word: hello"}], max_tokens=200)
    assert r.error is None, r.error
    assert "hello" in r.text.lower()
    assert r.usage["input_tokens"] > 0
    assert r.usage["output_tokens"] > 0

    r = llm.complete(
        [{"role": "user", "content": "Rate this pitch 1-5: 'A sea shanty about spreadsheets.'"}],
        schema=Verdict,
        max_tokens=300,
    )
    assert r.error is None, r.error
    assert isinstance(r.parsed, Verdict)
    assert r.structured_path in ("constrained", "parsed", "retried", "repaired")
    check_text_client(llm)

    dm = mk.decision(f"ollama:{name}", sink=sink)
    a = dm.decide("The sky is blue and it is a clear day.", {"blue": mk.YesNo("Is the sky blue?")})
    assert a["blue"]["error"] is None, a["blue"]["error"]
    assert a["blue"]["value"] > 0.5
    check_decision_client(dm)
    assert all(s["status"]["code"] == "ok" for s in sink.spans if s["name"] == "hone.models.chat")


def test_ac18_vision_on_a_generated_png(ollama_model, tmp_path: Path) -> None:
    name = ollama_model("HONE_TEST_VISION_MODEL", "qwen2.5vl:7b")
    llm = mk.text(f"ollama:{name}", sink=mk.records.MemorySink())
    image = red_png(tmp_path / "red.png")
    r = llm.complete(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the main color of this image? Answer with one word."},
                    {"type": "image", "path": str(image)},
                ],
            }
        ],
        max_tokens=50,
    )
    assert r.error is None, r.error
    assert "red" in r.text.lower()


def test_ac18_embeddings(ollama_model) -> None:
    name = ollama_model("HONE_TEST_EMBED_MODEL", "nomic-embed-text:latest")
    emb = mk.embedder(f"ollama:{name}", sink=mk.records.MemorySink())
    check_embedder(emb)
    rain, drizzle, taxes = emb.embed(["a song about rain", "a ballad about drizzle", "corporate tax law"])

    def cos(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cos(rain, drizzle) > cos(rain, taxes)
    assert math.isclose(cos(rain, rain), 1.0, rel_tol=1e-3)


def test_ac18_thinking_model(ollama_model) -> None:
    name = ollama_model("HONE_TEST_THINKING_MODEL", "deepseek-r1:8b")
    sink = mk.records.MemorySink()
    llm = mk.text(f"ollama:{name}", sink=sink)
    r = llm.complete(
        [{"role": "user", "content": "What is 2 + 3? Answer with the number only."}], max_tokens=300
    )
    assert llm.config.capabilities.thinking is True  # probed from /api/show, so think=false was sent
    assert r.error is None, r.error
    assert "5" in r.text
    assert "<think>" not in r.text
