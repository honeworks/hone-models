"""Images in the context budget (change 0013): header sizes, per-model cost, a context floor."""

import base64
import json
import struct
import zlib
from pathlib import Path

import pytest
import respx

import hone_models as mk
from hone_models._images import header_size, image_size
from hone_models.budget import DEFAULT_IMAGE_TOKENS, image_tokens, plan_context
from hone_models.registry import Capabilities, ModelConfig

URL = "http://127.0.0.1:11434"


def png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


def jpeg(width: int, height: int) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + sof + b"\xff\xd9"


def test_header_sizes_of_common_formats(tmp_path: Path) -> None:
    assert header_size(png(540, 960)) == (540, 960)
    assert header_size(jpeg(960, 540)) == (960, 540)
    assert header_size(b"GIF89a" + struct.pack("<HH", 30, 20)) == (30, 20)
    vp8x = (
        b"RIFF\x00\x00\x00\x00WEBPVP8X"
        + b"\x00" * 8
        + (99).to_bytes(3, "little")
        + (49).to_bytes(3, "little")
    )
    assert header_size(vp8x) == (100, 50)
    vp8l = b"RIFF\x00\x00\x00\x00WEBPVP8L" + b"\x00" * 5 + ((63) | (31 << 14)).to_bytes(4, "little")
    assert header_size(vp8l) == (64, 32)
    vp8 = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 10 + struct.pack("<HH", 320, 240)
    assert header_size(vp8) == (320, 240)
    assert header_size(b"RIFF\x00\x00\x00\x00WEBPXXXX") is None
    assert header_size(b"\xff\xd8\x00\x00") is None  # not a marker
    assert header_size(b"\x89PNG\r\n\x1a\n") is None  # truncated
    assert header_size(b"plain text") is None
    (tmp_path / "a.png").write_bytes(png(10, 20))
    assert image_size({"path": str(tmp_path / "a.png")}) == (10, 20)
    assert image_size({"data_b64": base64.b64encode(png(7, 8)).decode()}) == (7, 8)
    assert image_size({"path": str(tmp_path / "missing.png")}) is None
    assert image_size({"data_b64": "***"}) is None


def test_image_cost_per_model(tmp_path: Path) -> None:
    (tmp_path / "tall.png").write_bytes(png(540, 960))
    (tmp_path / "wide.jpg").write_bytes(jpeg(960, 540))
    parts = [{"type": "image", "path": str(tmp_path / n)} for n in ("tall.png", "wide.jpg")]
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Review."}, *parts]}]
    qwen = mk.registry.load().get("qwen2.5vl-7b")  # image_patch_px = 28: 20 x 35 tokens per image
    assert image_tokens(qwen, msgs) == 2 * 20 * 35
    flat = ModelConfig(id="f", provider="ollama", capabilities=Capabilities(image_tokens=256))
    assert image_tokens(flat, msgs) == 512
    unknown = ModelConfig(id="u", provider="ollama")
    assert image_tokens(unknown, msgs) == 2 * DEFAULT_IMAGE_TOKENS
    estimated, num_ctx, images = plan_context(qwen, msgs, 400)
    assert images == 1400
    assert estimated == 1400 + 2  # "Review." is 7 characters
    assert num_ctx == 2048  # 1802 * 1.1 rounded up: before, 2048 held only the text
    assert plan_context(qwen, msgs, 400, min_ctx=8192)[1] == 8192


def test_vision_call_sizes_num_ctx_for_its_images(tmp_path: Path) -> None:
    """The concept-shorts judge: two keyframes and max_tokens=400 overflowed a 2048 context."""
    (tmp_path / "a.png").write_bytes(png(540, 960))
    (tmp_path / "b.png").write_bytes(png(960, 540))
    content = [{"type": "text", "text": "Judge these frames. " * 40}]
    content += [{"type": "image", "path": str(tmp_path / n)} for n in ("a.png", "b.png")]
    sink = mk.records.MemorySink()
    reply = {"model": "m", "message": {"content": "ok"}, "done_reason": "stop"}
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat").respond(json=reply)
        mk.text("qwen2.5vl-7b", sink=sink).complete([{"role": "user", "content": content}], max_tokens=400)
    assert json.loads(chat.calls.last.request.content)["options"]["num_ctx"] == 4096
    attrs = sink.spans[-1]["attributes"]
    assert attrs["hone.models.context.estimated_image_tokens"] == 1400
    assert attrs["hone.models.context.estimated_prompt_tokens"] == 1400 + 229


def test_images_that_do_not_fit_overflow_before_the_call(tmp_path: Path) -> None:
    reg = mk.registry.Registry(
        {
            "tiny-vl": ModelConfig(
                id="tiny-vl", provider="ollama", capabilities=Capabilities(vision=True, max_input_tokens=2048)
            )
        }
    )
    (tmp_path / "a.png").write_bytes(png(64, 64))
    content = [{"type": "image", "path": str(tmp_path / "a.png")}] * 2
    with (
        respx.mock(assert_all_mocked=True),
        pytest.raises(mk.errors.ContextOverflow, match="2048 of them images"),
    ):
        mk.text("tiny-vl", registry=reg, sink=mk.records.NullSink()).complete(
            [{"role": "user", "content": content}], max_tokens=100
        )


def test_min_num_ctx_param_and_registry_default() -> None:
    reply = {"model": "m", "message": {"content": "ok"}, "done_reason": "stop"}
    with respx.mock(base_url=URL) as mock:
        chat = mock.post("/api/chat").respond(json=reply)
        llm = mk.text("gemma4-12b", sink=mk.records.NullSink())
        llm.complete([{"role": "user", "content": "hi"}], max_tokens=10, min_num_ctx=16384)
        sent = json.loads(chat.calls.last.request.content)
        assert sent["options"]["num_ctx"] == 16384
        assert "min_num_ctx" not in sent["options"]
        cfg = mk.registry.load().get("gemma4-12b").model_copy(update={"defaults": {"min_num_ctx": 8192}})
        llm = mk.text(
            "gemma4-12b", registry=mk.registry.Registry({"gemma4-12b": cfg}), sink=mk.records.NullSink()
        )
        llm.complete([{"role": "user", "content": "hi"}], max_tokens=10)
        assert json.loads(chat.calls.last.request.content)["options"]["num_ctx"] == 8192
    with pytest.raises(mk.errors.ConfigError, match="min_num_ctx must be a positive int"):
        mk.text("gemma4-12b", sink=mk.records.NullSink()).complete(
            [{"role": "user", "content": "x"}], min_num_ctx="big"
        )
