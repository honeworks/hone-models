"""Installed status and install commands (hone_models.catalog), and the packaged catalog files."""

import os
import subprocess
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

import hone_models as mk
from hone_models import catalog
from hone_models.errors import ConfigError, ProviderError
from hone_models.registry import ModelConfig
from hone_models.testing import FakeComfyUI, FakeOllama

DATA = Path(str(resources.files("hone_models"))) / "data" / "models"
COMFY = "http://127.0.0.1:8188"


def entry(**raw: Any) -> ModelConfig:
    return ModelConfig.model_validate({"id": "m", "provider": "comfyui", "kind": "image", **raw})


FILES = {"files": [{"repo": "org/r", "file": "a/b/x.safetensors", "to": "diffusion_models"}]}


def test_comfyui_answer_comes_from_the_server_without_a_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HONE_COMFYUI_DIR")  # ~/ComfyUI does not exist in the test HOME
    monkeypatch.delenv("HONE_COMFYUI_URL", raising=False)
    cfg = entry(install=FILES)
    info = {"UNETLoader": {"input": {"required": {"unet_name": [["x.safetensors", "y.gguf"], {}]}}}}
    with respx.mock(base_url=COMFY) as mock:
        route = mock.get("/object_info").respond(json=info)
        checker = catalog.Checker()
        assert checker.installed(cfg) == "yes"
        assert (
            checker.installed(entry(install={"files": [{"repo": "o/r", "file": "z", "to": "vae"}]})) == "no"
        )
        assert route.call_count == 1  # one listing asks the server once
        whole = entry(install={"files": [{"repo": "o/r", "to": "heartmula/X"}]})
        assert checker.installed(whole) == "unknown"  # a whole-repository folder is not listed
    with respx.mock(base_url=COMFY) as mock:
        mock.get("/object_info").mock(side_effect=httpx.ConnectError("refused"))
        assert catalog.installed(cfg) == "unknown"  # no answer: never "no" by guess


def test_comfyui_folder_named_but_missing_is_unknown() -> None:
    assert catalog.installed(entry(install=FILES)) == "unknown"  # HONE_COMFYUI_DIR points nowhere


def test_comfyui_folder_answers_by_file(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONE_COMFYUI_DIR", str(isolated / "comfy"))
    folder = isolated / "comfy" / "models" / "diffusion_models"
    folder.mkdir(parents=True)
    assert catalog.installed(entry(install=FILES)) == "no"
    (folder / "x.safetensors").write_bytes(b"w")
    assert catalog.installed(entry(install=FILES)) == "yes"
    whole = entry(install={"files": [{"repo": "o/r", "to": "heartmula/X"}]})
    assert catalog.installed(whole) == "no"
    (isolated / "comfy" / "models" / "heartmula" / "X").mkdir(parents=True)
    (isolated / "comfy" / "models" / "heartmula" / "X" / "config.json").write_text("{}")
    assert catalog.installed(whole) == "yes"


def test_project_folder_variable_and_check(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = entry(provider="command", kind="music", install={"dir_env": "HONE_P", "check": "test -f ok"})
    assert catalog.installed(cfg) == "unknown"  # the variable is unset
    monkeypatch.setenv("HONE_P", str(isolated / "p"))
    assert catalog.installed(cfg) == "no"  # no such folder
    (isolated / "p").mkdir()
    assert catalog.installed(cfg) == "no"  # the check fails
    (isolated / "p" / "ok").write_text("")
    assert catalog.installed(cfg) == "yes"
    no_check = entry(provider="command", kind="music", install={"dir_env": "HONE_P"})
    assert catalog.installed(no_check) == "yes"
    broken = entry(provider="command", kind="music", install={"dir_env": "HONE_P", "check": "no-such-cmd-x"})
    assert catalog.installed(broken) == "no"


def test_ollama_names_hosted_and_bare_entries() -> None:
    def tags(path: str, body: dict[str, Any]) -> dict[str, Any] | None:
        return (
            {"models": [{"name": "plain:latest"}, {"name": "hf.co/o/R-GGUF:Q4_K_M"}]}
            if path == "/api/tags"
            else None
        )

    with FakeOllama(tags):
        assert catalog.installed(ModelConfig(id="p", provider="ollama", model="plain")) == "yes"  # `:latest`
        tagged = ModelConfig.model_validate(
            {"id": "t", "provider": "ollama", "install": {"ollama": "hf.co/o/R-GGUF:Q4_K_M"}}
        )
        assert catalog.installed(tagged) == "yes"
        assert catalog.installed(ModelConfig(id="q", provider="ollama", model="other:7b")) == "no"
    hosted = ModelConfig(id="h", provider="openai_compatible", base_url="https://api.openai.com/v1")
    assert catalog.installed(hosted) == "yes"  # nothing to install
    assert catalog.installed(entry()) == "unknown"  # no install table


def test_hf_cache_location(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(isolated / "hfhome"))
    assert catalog.hf_cache() == isolated / "hfhome" / "hub"
    monkeypatch.delenv("HF_HOME")
    assert catalog.hf_cache() == Path.home() / ".cache" / "huggingface" / "hub"
    cfg = entry(provider="faster_whisper", kind="transcription", install={"hf": "o/w"})
    assert catalog.installed(cfg) == "no"
    snapshot = catalog.hf_cache() / "models--o--w" / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    assert catalog.installed(cfg) == "no"  # an empty snapshot is not a download
    (snapshot / "model.bin").write_bytes(b"w")
    assert catalog.installed(cfg) == "yes"


def test_install_commands_and_folders(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    whole = entry(install={"files": [{"repo": "o/r", "to": "heartmula/X"}], "hf": "o/extra", "size_gb": 1.0})
    target = catalog.comfyui_dir() / "models" / "heartmula" / "X"
    assert catalog.install_commands(whole) == [f"hf download o/r --local-dir {target}", "hf download o/extra"]
    assert catalog.install_folder(whole) == catalog.comfyui_dir() / "models"
    assert catalog.install_commands(entry()) == []
    project = entry(
        provider="command",
        kind="music",
        install={"repo": "https://x/p.git", "setup": ["make"], "dir_env": "HONE_P"},
    )
    assert catalog.install_commands(project) == ["git clone https://x/p.git $HONE_P", "cd $HONE_P && make"]
    assert catalog.install_folder(project) == Path.home()
    monkeypatch.setenv("HONE_P", str(isolated / "my project"))
    assert catalog.install_commands(project)[0] == f"git clone https://x/p.git '{isolated / 'my project'}'"
    assert catalog.install_folder(project) == isolated / "my project"
    ollama = ModelConfig.model_validate({"id": "o", "provider": "ollama", "install": {"ollama": "a:1b"}})
    monkeypatch.setenv("OLLAMA_MODELS", str(isolated / "om"))
    assert catalog.install_folder(ollama) == isolated / "om"
    assert catalog.install_folder(entry(install={"hf": "o/w"})) == catalog.hf_cache()
    assert catalog.free_gb(isolated / "does" / "not" / "exist") is not None  # the nearest existing parent


def test_run_install_runs_each_command_and_stops_at_a_failure() -> None:
    ran: list[str] = []

    def fake(command: str, *, shell: bool, check: bool) -> None:
        ran.append(command)
        if "fail" in command:
            raise subprocess.CalledProcessError(3, command)

    cfg = entry(install={"hf": "o/w", "ollama": "fail:1b"})
    with pytest.raises(ProviderError, match=r"'ollama pull fail:1b' exited 3"):
        catalog.run_install(cfg, run=fake)
    assert ran == ["ollama pull fail:1b"]
    assert catalog.run_install(entry(install={"hf": "o/w"}), run=fake) == ["hf download o/w"]
    with pytest.raises(ConfigError, match="no install commands; see its source: https://s"):
        catalog.run_install(entry(install={"source": "https://s"}), run=fake)


def test_require_installed_passes_unknown() -> None:
    catalog.require_installed(entry())  # unknown: the call goes ahead


# --- the packaged catalog files ------------------------------------------------------------------


def test_every_packaged_entry_is_valid_and_complete() -> None:
    reg = mk.registry.load()
    for path in sorted(DATA.glob("*.toml")):
        models = tomllib.loads(path.read_text(encoding="utf-8"))["models"]
        for model_id in models:
            cfg = reg.models[model_id]
            assert cfg.kind == path.stem, f"{model_id} is in {path.name}"
            assert cfg.install is not None, model_id
            assert cfg.capabilities.license, model_id
            assert cfg.guide is not None, model_id
            assert cfg.guide.summary, model_id
            assert cfg.install.source or cfg.install.ollama, model_id
    assert len(reg.models) >= 80  # the catalog, not only the eight original defaults
    originals = ("gemma4-12b", "qwen2.5vl-7b", "deepseek-r1-8b", "nomic-embed-text", "jev", "gpt-4.1-mini")
    assert set(originals) | {"kokoro-82m", "chatterbox"} <= set(reg.models)


def test_packaged_catalog_has_no_machine_paths_and_marks_non_commercial_models() -> None:
    text = "\n".join(p.read_text(encoding="utf-8") for p in DATA.glob("*.toml"))
    assert "/home/" not in text
    assert '"~/' not in text  # no home paths in values (comments may name ~/ComfyUI)
    reg = mk.registry.load()
    non_commercial = (
        "hemmingway-1",
        "yue2-3b",
        "flux.2-klein-9b",
        "flux.2-dev",
        "qwen-image-2.1",
        "mulacover",
    )
    for model_id in (*non_commercial, "songgeneration-v2-medium", "muq-mulan-large"):
        assert reg.models[model_id].capabilities.commercial_use is False, model_id
    assert reg.models["qwen-image-edit-2511"].capabilities.features == [
        "camera angle",
        "multi-image composition",
    ]


def test_scoring_entries_cannot_be_called() -> None:
    with pytest.raises(ConfigError, match="no client for kind 'scoring' yet"):
        mk.speech("htdemucs")
    with pytest.raises(ConfigError, match="no client for kind 'scoring' yet"):
        mk.embedder("dinov2-small")


def test_comfyui_catalog_entry_without_workflow_says_so(isolated: Path) -> None:
    with FakeComfyUI(), pytest.raises(ConfigError, match=r"no workflow yet for 'flux\.2-klein-4b'"):
        mk.image("flux.2-klein-4b").generate("a cat", out=isolated / "cat.png")


def test_a_packaged_id_declared_twice_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    folder = tmp_path / "hone_models" / "data" / "models"
    folder.mkdir(parents=True)
    for name in ("a.toml", "b.toml"):
        (folder / name).write_text('[models.same]\nprovider = "ollama"\n')
    (folder / "notes.txt").write_text("not a registry file")
    monkeypatch.setattr("hone_models.registry.resources.files", lambda package: tmp_path / "hone_models")
    with pytest.raises(ConfigError, match=r"\['same'\] declared again in data/models/b.toml"):
        mk.registry.load()
    assert os.environ.get("HONE_COMFYUI_DIR")  # the test isolation keeps installed checks offline
