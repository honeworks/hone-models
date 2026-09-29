"""An entry marked `disabled` is known (listed, guided) but every call refuses with its reason."""

import pytest

import hone_models as mk
from hone_models.errors import ConfigError
from hone_models.registry import ModelConfig, Registry, require_client


def test_a_disabled_entry_refuses_every_call_with_its_reason() -> None:
    cfg = ModelConfig(id="broken", provider="ollama", disabled="its weights do not load yet (TODO)")
    with pytest.raises(
        ConfigError, match=r"model 'broken' is disabled: its weights do not load yet \(TODO\)"
    ):
        require_client(cfg)
    require_client(ModelConfig(id="fine", provider="ollama"))  # no reason, no error


def test_selection_skips_disabled_entries() -> None:
    reg = Registry({
        "a": ModelConfig(id="a", provider="ollama", disabled="not yet"),
        "b": ModelConfig(id="b", provider="ollama"),
    })  # fmt: skip
    assert [m.id for m in reg.matching()] == ["b"]


def test_the_packaged_songgeneration_entry_is_disabled_until_its_runtime_matches() -> None:
    cfg = mk.registry.load().get("songgeneration-v2-medium")
    assert cfg.disabled is not None
    assert "D-080" in cfg.disabled
    with pytest.raises(ConfigError, match="songgeneration-v2-medium' is disabled"):
        mk.music("songgeneration-v2-medium")
