"""AC-6: require={"vision": True} with prefer="local" picks the local vision model; else lists candidates."""

import pytest

import hone_models as mk

pytestmark = pytest.mark.e2e


def test_ac6_picks_local_vision_model() -> None:
    vision = mk.text(require={"vision": True, "min_context": 8000}, prefer="local")
    assert vision.model_id == "qwen2.5vl-7b"
    assert vision.config.local is True
    hosted = mk.text(require={"vision": True}, prefer="hosted")
    assert hosted.model_id == "gpt-4.1-mini"


def test_ac6_no_match_lists_candidates() -> None:
    with pytest.raises(mk.errors.CapabilityError) as info:
        mk.text(require={"vision": True, "min_context": 2_000_000}, prefer="local")
    message = str(info.value)
    assert "closest candidates" in message
    assert "gpt-4.1-mini (lacks min_context=2000000)" in message
    assert "qwen2.5vl-7b" in message
