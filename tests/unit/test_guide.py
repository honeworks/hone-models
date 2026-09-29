"""Model guides (hone_models.guide) and the `models guide` / `models install` / `models list` commands."""

import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import hone_models as mk
from hone_models.cli import app
from hone_models.guide import build, stale
from hone_models.registry import ModelConfig, Registry
from hone_models.testing import FakeOllama

runner = CliRunner()
CATALOG = str(Path(__file__).parents[1] / "fixtures" / "catalog" / "models.toml")


def test_every_packaged_guide_has_a_text_and_a_json_form() -> None:
    for cfg in mk.registry.load().models.values():
        g = build(cfg, installed="unknown")  # no machine check here
        assert g.as_text().startswith(f"{cfg.id} ({cfg.kind})")
        assert json.loads(json.dumps(g.as_dict()))["id"] == cfg.id


def test_guide_of_hosted_and_listed_inputs() -> None:
    video = build(mk.registry.load().get("sora-2"), installed="yes")
    text = video.as_text()
    assert "sizes 720x1280, 1280x720" in text
    assert "durations 4, 8, 12 s" in text
    assert "Installed: yes" in text
    assert "install:" not in text  # nothing to install for a hosted model
    song = build(mk.registry.load().get("songgeneration-v2-medium"), installed="no")
    assert song.inputs["generate_type"].origin == "own"
    assert "LeVo" in (song.inputs["lyrics"].note or "")  # the entry's own note wins
    assert "at most 270 s" in song.as_text()
    assert "install: hone-models models install songgeneration-v2-medium" in song.as_text()
    chat = ModelConfig.model_validate({"id": "c", "provider": "command", "kind": "chat", "inputs": ["mode"]})
    assert set(build(chat, installed="unknown").inputs) == {"mode"}
    music = ModelConfig.model_validate(
        {"id": "s", "provider": "command", "kind": "music", "lyrics_format": "plain"}
    )
    lyrics = build(music, installed="unknown").inputs["lyrics"]
    assert lyrics.note is not None
    assert lyrics.note.endswith("converted to the model's 'plain' form")
    bare = ModelConfig.model_validate(
        {"id": "b", "provider": "none", "kind": "scoring", "capabilities": {"commercial_use": False}}
    )
    assert "License: not stated (commercial use: not allowed)" in build(bare, installed="unknown").as_text()


def test_stale_lists_unchecked_and_old_guides_oldest_first() -> None:
    def cfg(model_id: str, checked: str | None) -> ModelConfig:
        guide = {"summary": "s", "checked": checked} if checked else None
        return ModelConfig.model_validate({"id": model_id, "provider": "ollama", "guide": guide})

    reg = Registry({m.id: m for m in (cfg("new", "2026-09-20"), cfg("old", "2026-01-01"), cfg("none", None))})
    assert stale(reg, 90, today=date(2026, 9, 29)) == [("none", None), ("old", date(2026, 1, 1))]
    assert stale(reg, 1, today=date(2026, 9, 29))[-1] == ("new", date(2026, 9, 20))
    assert stale(mk.registry.load(), 10_000) == [("jev", None)]  # its API shape is unverified (D-019)
    assert "gemma4-12b" in dict(stale(mk.registry.load(), 0, today=date(2100, 1, 1)))


def test_cli_guide_stale_and_errors() -> None:
    result = runner.invoke(app, ["models", "guide", "--stale", "90", "--registry", CATALOG])
    assert result.exit_code == 0, result.output
    assert "cat-bare: never checked" in result.output
    as_json = runner.invoke(
        app, ["models", "guide", "cat-chat-here", "--stale", "90", "--json", "--registry", CATALOG]
    )
    assert json.loads(as_json.output) == [{"id": "cat-chat-here", "checked": None}]
    assert runner.invoke(app, ["models", "guide"]).exit_code == 1  # an id or --stale
    missing = runner.invoke(app, ["models", "guide", "nope"])
    assert missing.exit_code == 1
    assert "unknown model" in missing.stderr
    with FakeOllama():
        text = runner.invoke(app, ["models", "guide", "gemma4-12b"])
    assert text.exit_code == 0
    assert "Source: https://huggingface.co/google/gemma-4-12B-it (checked 2026-09-29)" in text.output


def test_cli_install_run_executes_the_printed_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []

    def fake(command: str, **kwargs: Any) -> None:
        ran.append(command)

    monkeypatch.setattr(subprocess, "run", fake)
    result = runner.invoke(app, ["models", "install", "cat-whisper", "--run", "--registry", CATALOG])
    assert result.exit_code == 0, result.output
    assert ran == ["hf download org/whisper-x"]
    assert "# size unknown GB" in result.output
    hosted = runner.invoke(app, ["models", "install", "sora-2"])
    assert "# nothing to download: see the source" in hosted.output
    assert "# hosted: nothing to install; set OPENAI_API_KEY" in hosted.output
    failed = runner.invoke(app, ["models", "install", "sora-2", "--run"])
    assert failed.exit_code == 1
    assert "no install commands" in failed.stderr


def test_cli_list_table_shows_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    with FakeOllama():
        result = runner.invoke(app, ["models", "list", "--kind", "embedding"], env={"COLUMNS": "120"})
    assert result.exit_code == 0, result.output
    assert "nomic-embed-text" in result.output
    assert "installed" in result.output
    assert " no " in result.output  # FakeOllama lists no models
