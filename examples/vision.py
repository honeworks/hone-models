"""Vision: send images to a model that can see them, and refuse models that cannot, before calling.

What: pick a vision model by capability, send an image from a file or from memory, and see the
    `CapabilityError` a text-only model gives instead of a confusing HTTP 400.

How: a message's `content` becomes a list of parts: `{"type": "text", "text": ...}`,
    `{"type": "image", "path": "frame.png"}` or `{"type": "image", "data_b64": ..., "mime": "image/png"}`.
    File images are read and inlined when the call is made. `mk.text(require={"vision": True})` chooses
    a model whose registry entry says `vision = true`.

Why: a model that cannot read images either fails with an unhelpful server error or ignores the image
    and answers anyway. Here the registry's `vision` capability is checked first: `vision = false` raises
    `mk.errors.CapabilityError` and nothing is sent. Pitfall: an ad-hoc id's `vision` is unknown (not
    checked) unless the provider can be probed (`ollama:`, `litellm:`); register the models you use.
"""

import base64
import tempfile
from pathlib import Path

import hone_models as mk
from hone_models.testing import FakeOllama

PNG_1X1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

server = FakeOllama().start()  # offline stand-in for Ollama (sets OLLAMA_HOST)
frame = Path(tempfile.mkdtemp()) / "frame.png"
frame.write_bytes(base64.b64decode(PNG_1X1))

# 1. Choose any local model that can see, and send a file image.
vision = mk.text(require={"vision": True}, prefer="local", sink=mk.records.MemorySink())
r = vision.complete(
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in one sentence."},
                {"type": "image", "path": str(frame)},
            ],
        }
    ]
)
sent = server.requests[-1]["body"]["messages"][0]
print("model:", vision.model_id, "| images sent:", len(sent["images"]), "| reply:", r.text)
assert vision.model_id == "qwen2.5vl-7b"
assert sent["images"] == [PNG_1X1]  # inlined as base64

# 2. An image already in memory (e.g. a rendered frame) goes in as base64 with its MIME type.
vision.complete(
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What colour is it?"},
                {"type": "image", "data_b64": PNG_1X1, "mime": "image/png"},
            ],
        }
    ]
)
assert server.requests[-1]["body"]["messages"][0]["images"] == [PNG_1X1]

# 3. A text-only model is refused before any request.
calls_before = len(server.requests)
try:
    mk.text("gemma4-12b", sink=mk.records.MemorySink()).complete(
        [{"role": "user", "content": [{"type": "image", "path": str(frame)}]}]
    )
except mk.errors.CapabilityError as exc:
    print("refused:", exc)
else:
    raise AssertionError("gemma4-12b is declared without vision")
assert len(server.requests) == calls_before

server.stop()
