"""AC-16: registry merge + ad-hoc ids + `models list/show` CLI with a --json schema."""

import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

import hone_models as mk
from hone_models.cli import app
from hone_models.registry import GENERATION_KEYS, Capabilities
from hone_models.testing import FakeOllama

pytestmark = pytest.mark.e2e
runner = CliRunner()
MODEL_KEYS = {
    "id", "provider", "model", "name", "kind", "base_url", "base_url_env", "api_key_env",
    "defaults", "capabilities", "max_timeout_s", "local",
}  # fmt: skip


@pytest.fixture(autouse=True)
def ollama() -> Iterator[FakeOllama]:
    """`models list` asks Ollama which models are installed: a fake answers (also for the subprocess)."""
    with FakeOllama() as server:
        yield server


def test_ac16_registry_merge_and_adhoc(isolated: Path) -> None:
    user = isolated / "home/.config/hone/models.toml"
    user.parent.mkdir(parents=True)
    user.write_text('[models.mine]\nprovider = "ollama"\nmodel = "llama3.2:1b"\n')
    (isolated / "hone-models.toml").write_text("[models.mine.capabilities]\nvision = false\n")
    reg = mk.registry.load()
    assert reg.get("mine").name == "llama3.2:1b"
    assert reg.get("mine").capabilities.vision is False
    assert "gemma4-12b" in reg.models  # packaged defaults still there
    assert reg.get("ollama:qwen3:4b").provider == "ollama"


def test_ac16_models_list_json(isolated: Path) -> None:
    extra = isolated / "extra.toml"
    extra.write_text('[models.extra]\nprovider = "openai_compatible"\nbase_url = "http://localhost:1/v1"\n')
    result = runner.invoke(app, ["models", "list", "--json", "--registry", str(extra)])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    # the generation keys (change 0015) only where an entry sets them; `installed` on every row
    assert all(
        MODEL_KEYS | {"installed"} <= set(r) <= MODEL_KEYS | {"installed", *GENERATION_KEYS} for r in rows
    )
    assert (
        set(
            json.loads(
                runner.invoke(app, ["models", "show", "extra", "--json", "--registry", str(extra)]).output
            )
        )
        == MODEL_KEYS
    )
    by_id = {r["id"]: r for r in rows}
    assert by_id["extra"]["local"] is True
    assert by_id["gemma4-12b"]["capabilities"]["max_input_tokens"] == 32768
    assert all(set(r["capabilities"]) == set(Capabilities.model_fields) for r in rows)
    assert by_id["gpt-4.1-mini"]["capabilities"]["price"] == {
        "input_per_mtok": 0.4,
        "output_per_mtok": 1.6,
        "per_image": 0.0,
        "per_output_second": 0.0,
    }
    assert [r["id"] for r in rows] == sorted(by_id)


def test_ac16_models_show() -> None:
    result = runner.invoke(app, ["models", "show", "ollama:llama3.2:1b", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert set(data) == MODEL_KEYS
    assert (data["provider"], data["name"]) == ("ollama", "llama3.2:1b")
    human = runner.invoke(app, ["models", "show", "gemma4-12b"])
    assert "gemma4-12b:latest" in human.output
    table = runner.invoke(app, ["models", "list"])
    assert table.exit_code == 0
    assert "gemma4-12b" in table.output


def test_ac16_cli_errors_exit_1(isolated: Path) -> None:
    result = runner.invoke(app, ["models", "show", "nope", "--json"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "unknown model" in result.stderr
    bad = isolated / "bad.toml"
    bad.write_text("[models\n")
    result = runner.invoke(app, ["models", "list", "--registry", str(bad)])
    assert result.exit_code == 1
    assert "not valid TOML" in result.stderr
    result = runner.invoke(app, ["models", "list", "--registry", str(isolated / "missing.toml")])
    assert result.exit_code == 1
    assert runner.invoke(app, ["--help"]).exit_code == 0


def test_ac16_installed_script_runs() -> None:
    script = shutil.which("hone-models", path=str(Path(sys.executable).parent))
    assert script, "the hone-models console script is not installed"
    out = subprocess.run([script, "models", "list", "--json"], capture_output=True, text=True, check=True)
    assert out.stderr == ""
    assert {r["id"] for r in json.loads(out.stdout)} >= {"gemma4-12b", "jev"}
    human = subprocess.run([script, "models", "list"], capture_output=True, text=True, check=True)
    assert "\x1b[" not in human.stdout  # no colour codes when not a TTY
