"""`capabilities.features` from the guide, and `require={"features": [...]}` (change 0015 §3a)."""

import pytest

import hone_models as mk
from hone_models.errors import CapabilityError
from hone_models.registry import Capabilities, ModelConfig, Registry, unmet

GUIDE = {"summary": "s", "features": [{"name": "Camera angle"}, {"name": "text"}]}


def test_features_come_from_the_guide_unless_declared() -> None:
    cfg = ModelConfig.model_validate({"id": "a", "provider": "comfyui", "kind": "image", "guide": GUIDE})
    assert cfg.capabilities.features == ["Camera angle", "text"]
    own = {
        "id": "b",
        "provider": "comfyui",
        "kind": "image",
        "guide": GUIDE,
        "capabilities": {"features": ["x"]},
    }
    assert ModelConfig.model_validate(own).capabilities.features == ["x"]
    built = ModelConfig(
        id="c", provider="comfyui", kind="image", guide=cfg.guide, capabilities=Capabilities(vram_gb=2.0)
    )
    assert (built.capabilities.features, built.capabilities.vram_gb) == (["Camera angle", "text"], 2.0)


def test_features_requirement_needs_every_feature_case_ignored() -> None:
    cfg = ModelConfig.model_validate({"id": "a", "provider": "comfyui", "kind": "image", "guide": GUIDE})
    assert unmet(cfg, {"features": ["camera angle"]}) == []
    assert unmet(cfg, {"features": "TEXT"}) == []  # one name as a string
    assert unmet(cfg, {"features": ["text", "lyrics"]}) == ["features ∋ 'lyrics'"]
    plain = ModelConfig(id="p", provider="comfyui", kind="image")
    reg = Registry({"a": cfg, "p": plain})
    assert [m.id for m in mk.select({"features": ["text"]}, kind="image", registry=reg)] == ["a"]
    assert mk.select({"features": ["lyrics"]}, kind="image", registry=reg) == []
    with pytest.raises(CapabilityError, match="lacks features"):
        reg.select({"features": ["lyrics"]}, kind="image")
