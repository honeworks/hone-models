"""AC-5: an image sent to a non-vision model raises CapabilityError before calling."""

import pytest
import respx

import hone_models as mk

pytestmark = pytest.mark.e2e


def test_ac5_image_to_non_vision_model(isolated) -> None:
    (isolated / "frame.png").write_bytes(b"\x89PNG")
    llm = mk.text("gemma4-12b", sink=mk.records.NullSink())  # vision = false in the registry
    message = {
        "role": "user",
        "content": [{"type": "text", "text": "Describe."}, {"type": "image", "path": "frame.png"}],
    }
    with respx.mock(assert_all_mocked=True) as mock:
        with pytest.raises(
            mk.errors.CapabilityError, match=r"cannot read images.*require=\{'vision': True\}"
        ):
            llm.complete([message])
        assert mock.calls.call_count == 0


def test_ac5_probed_adhoc_model_without_vision(recorded) -> None:
    sink = mk.records.MemorySink()
    message = {"role": "user", "content": [{"type": "image", "data_b64": "QUJD", "mime": "image/png"}]}
    with respx.mock(base_url="http://127.0.0.1:11434", assert_all_called=False) as mock:
        show = mock.post("/api/show").respond(json=recorded("ollama_show_llama.json"))
        chat = mock.post("/api/chat")
        with pytest.raises(mk.errors.CapabilityError):
            mk.text("ollama:llama3.2:1b", sink=sink).complete([message])
    assert show.call_count == 1
    assert chat.call_count == 0
    assert sink.spans[0]["status"]["code"] == "error"
