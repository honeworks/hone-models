from pathlib import Path

import pytest

from hone_models.errors import CapabilityError, ConfigError
from hone_models.registry import Registry, load


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_packaged_defaults_load() -> None:
    reg = load()
    gemma = reg.get("gemma4-12b")
    assert gemma.provider == "ollama"
    assert gemma.name == "gemma4-12b:latest"
    assert gemma.capabilities.thinking is True
    assert gemma.capabilities.max_input_tokens == 32768
    assert gemma.defaults == {"temperature": 0.8}
    assert reg.get("nomic-embed-text").kind == "embedding"
    assert reg.get("jev").kind == "decision"


def test_merge_order_user_project_explicit(isolated: Path) -> None:
    write(isolated / "home/.config/hone/models.toml", '[models."gemma4-12b".capabilities]\nvram_gb = 1.0\n')
    write(isolated / "hone-models.toml", '[models."gemma4-12b"]\ndefaults = { temperature = 0.1 }\n')
    extra = write(isolated / "extra.toml", '[models."gemma4-12b".capabilities]\nmax_input_tokens = 999\n')
    cfg = load([extra]).get("gemma4-12b")
    assert cfg.capabilities.vram_gb == 1.0  # user file
    assert cfg.defaults == {"temperature": 0.1}  # project file
    assert cfg.capabilities.max_input_tokens == 999  # explicit path, last
    assert cfg.capabilities.thinking is True  # untouched packaged value survives the merge


def test_missing_explicit_path_is_an_error(isolated: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load([isolated / "nope.toml"])


def test_invalid_toml_and_unknown_fields(isolated: Path) -> None:
    with pytest.raises(ConfigError, match="not valid TOML"):
        load([write(isolated / "bad.toml", "[models\n")])
    with pytest.raises(ConfigError, match="invalid registry entry for 'x'"):
        load([write(isolated / "typo.toml", '[models.x]\nprovider = "ollama"\nvisoin = true\n')])
    with pytest.raises(ConfigError, match="invalid registry entry"):
        load([write(isolated / "prov.toml", '[models.x]\nprovider = "nope"\n')])


def test_get_by_provider_model_name() -> None:
    assert load().get("qwen2.5vl:7b").id == "qwen2.5vl-7b"


def test_adhoc_ids() -> None:
    reg = load()
    ollama = reg.get("ollama:llama3.2:1b")
    assert (ollama.provider, ollama.name, ollama.local) == ("ollama", "llama3.2:1b", True)
    assert ollama.capabilities.vision is None  # unknown until probed
    openai = reg.get("openai:gpt-4.1")
    assert openai.provider == "openai_compatible"
    assert openai.base_url == "https://api.openai.com/v1"
    assert openai.api_key_env == "OPENAI_API_KEY"
    assert openai.local is False


def test_unknown_id_lists_registered_ids() -> None:
    with pytest.raises(ConfigError, match=r"unknown model 'nope'.*gemma4-12b.*<provider>:<model>"):
        load().get("nope")
    with pytest.raises(ConfigError):
        load().get("bogus:thing")


def test_local_detection_for_openai_compatible(isolated: Path) -> None:
    path = write(
        isolated / "r.toml",
        '[models.lcpp]\nprovider = "openai_compatible"\nbase_url = "http://127.0.0.1:8080/v1"\n'
        '[models.remote]\nprovider = "openai_compatible"\n',
    )
    reg = load([path])
    assert reg.get("lcpp").local is True
    assert reg.get("remote").local is False
    assert reg.get("jev").local is False


def reg_for_select(isolated: Path) -> Registry:
    path = write(
        isolated / "sel.toml",
        """
[models.local_small]
provider = "ollama"
[models.local_small.capabilities]
vision = true
max_input_tokens = 4096
speed_tok_s = 80.0
[models.local_big]
provider = "ollama"
[models.local_big.capabilities]
vision = true
max_input_tokens = 32768
speed_tok_s = 20.0
[models.hosted_cheap]
provider = "openai_compatible"
base_url = "https://example.com/v1"
[models.hosted_cheap.capabilities]
vision = true
max_input_tokens = 128000
price = { input_per_mtok = 0.1, output_per_mtok = 0.2 }
""",
    )
    reg = load([path])
    return Registry(
        {k: v for k, v in reg.models.items() if k in {"local_small", "local_big", "hosted_cheap"}}
    )


def test_select_by_capability_and_preference(isolated: Path) -> None:
    reg = reg_for_select(isolated)
    assert reg.select({"vision": True, "min_context": 8000}, prefer="local").id == "local_big"
    assert reg.select({"vision": True, "min_context": 8000}, prefer="hosted").id == "hosted_cheap"
    assert reg.select({"vision": True}, prefer="cheapest").id == "local_big"  # local costs 0; ties by id
    assert reg.select({"vision": True}, prefer="fastest").id == "local_small"
    assert reg.select().id == "hosted_cheap"  # no preference: first id


def test_select_no_match_lists_closest(isolated: Path) -> None:
    reg = reg_for_select(isolated)
    with pytest.raises(
        CapabilityError, match=r"closest candidates: hosted_cheap \(lacks min_context=999999\)"
    ):
        reg.select({"vision": True, "min_context": 999999})
    with pytest.raises(CapabilityError, match="no embedding model"):
        reg.select(kind="embedding")


def test_select_rejects_unknown_requirement_and_preference(isolated: Path) -> None:
    reg = reg_for_select(isolated)
    with pytest.raises(ConfigError, match="unknown requirement 'wings'"):
        reg.select({"wings": True})
    with pytest.raises(ConfigError, match="unknown prefer='soonest'"):
        reg.select(prefer="soonest")


def test_entry_errors_are_typed(isolated: Path) -> None:
    with pytest.raises(ConfigError, match="remove the 'id' key"):
        load([write(isolated / "id.toml", '[models.x]\nprovider = "ollama"\nid = "y"\n')])
    with pytest.raises(ConfigError, match=r"unknown top-level keys \['model'\]"):
        load([write(isolated / "typo.toml", '[model.x]\nprovider = "ollama"\n')])
    with pytest.raises(ConfigError, match="min_context must be an int"):
        load().select({"min_context": "8000"})


def test_load_accepts_a_single_path_string(isolated: Path) -> None:
    path = write(isolated / "one.toml", '[models.one]\nprovider = "ollama"\n')
    assert "one" in load(str(path)).models


def test_later_list_replaces_and_tables_merge(isolated: Path) -> None:
    path = write(
        isolated / "m.toml",
        '[models.jev.capabilities]\nquestions = ["yes_no"]\n[models."gemma4-12b".defaults]\ntop_p = 0.9\n',
    )
    reg = load([path])
    assert reg.get("jev").capabilities.questions == ["yes_no"]
    assert reg.get("gemma4-12b").defaults == {"temperature": 0.8, "top_p": 0.9}


def test_adhoc_edge_cases() -> None:
    reg = load()
    with pytest.raises(ConfigError):
        reg.get("ollama:")
    assert reg.get("jev:x").kind == "decision"
    lite = reg.get("litellm:anthropic/claude")
    assert (lite.provider, lite.name, lite.local) == ("litellm", "anthropic/claude", False)


@pytest.mark.parametrize("url", ["http://localhost:8080/v1", "http://[::1]:8080/v1"])
def test_localhost_variants_are_local(isolated: Path, url: str) -> None:
    path = write(isolated / "l.toml", f'[models.l]\nprovider = "openai_compatible"\nbase_url = "{url}"\n')
    assert load([path]).get("l").local is True


def test_select_unknown_values_sort_last_and_unmet_capabilities(isolated: Path) -> None:
    path = write(
        isolated / "u.toml",
        '[models.a]\nprovider = "openai_compatible"\n'
        '[models.b]\nprovider = "openai_compatible"\n[models.b.capabilities]\nspeed_tok_s = 0.5\n'
        "price = { input_per_mtok = 5.0 }\n",
    )
    reg = Registry({k: v for k, v in load([path]).models.items() if k in {"a", "b"}})
    assert reg.select(prefer="fastest").id == "b"  # a's speed is unknown
    assert reg.select(prefer="cheapest").id == "b"  # a's price is unknown (inf)
    with pytest.raises(CapabilityError, match=r"a \(lacks thinking=True, min_context=10\)"):
        reg.select({"thinking": True, "min_context": 10})


def test_select_embedding_from_defaults() -> None:
    assert load().select(kind="embedding").id == "nomic-embed-text"
