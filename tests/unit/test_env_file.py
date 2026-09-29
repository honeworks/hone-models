"""`.env` parsing and loading (change 0017)."""

import os
from pathlib import Path

import pytest

import hone_models as mk
from hone_models import records
from hone_models._env_file import apply_env_files, load_env_file
from hone_models.errors import ConfigError
from hone_models.registry import ModelConfig, Registry, require_client


def write(folder: Path, text: str, name: str = ".env") -> Path:
    path = folder / name
    path.write_text(text)
    return path


def test_parses_comments_export_and_quotes(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "# comment\n\n  export A=1\nB = two words # a comment\nC=\"x # y\"\nD='it''s'\nE=\nF=a#b\nA=last\n",
    )
    assert load_env_file(path) == {
        "A": "last",
        "B": "two words",
        "C": "x # y",
        "D": "it''s",
        "E": "",
        "F": "a#b",
    }


@pytest.mark.parametrize(
    "line", ["no-equals-sign", "1BAD=x", "BAD KEY=x", "=x", 'Q="unclosed', "Q='a' tail", "export"]
)
def test_a_malformed_line_names_the_line_not_its_text(tmp_path: Path, line: str) -> None:
    path = write(tmp_path, f"OK=1\n{line}\n")
    with pytest.raises(ConfigError, match=r"line 2") as caught:
        load_env_file(path)
    assert str(path) in str(caught.value)
    assert line not in str(caught.value)


def test_a_file_that_is_not_utf8(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes(b"A=\xff\n")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_env_file(path)


def test_each_file_is_read_once_and_never_overwrites(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ENVF_A", "ENVF_B"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENVF_B", "shell")
    path = write(isolated, "ENVF_A=first\nENVF_B=file\n")
    apply_env_files()
    assert (os.environ["ENVF_A"], os.environ["ENVF_B"]) == ("first", "shell")
    monkeypatch.delenv("ENVF_A")
    path.write_text("ENVF_A=second\n")
    apply_env_files()  # already read in this process
    assert "ENVF_A" not in os.environ


def test_no_dot_env_is_fine(isolated: Path) -> None:
    apply_env_files()


def judge(base_url: str | None = None) -> ModelConfig:
    return ModelConfig(id="j", provider="openai_compatible", base_url=base_url, base_url_env="ENVF_URL")


def test_base_url_env_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENVF_URL", raising=False)
    unset = judge()
    assert unset.base_url is None
    with pytest.raises(ConfigError, match="ENVF_URL"):
        require_client(unset)
    assert judge("http://localhost:1/v1").base_url == "http://localhost:1/v1"
    monkeypatch.setenv("ENVF_URL", "http://127.0.0.1:2/v1")
    resolved = judge("http://localhost:1/v1")
    assert resolved.base_url == "http://127.0.0.1:2/v1"
    assert resolved.local is True
    require_client(resolved)


def test_url_credentials_are_scrubbed() -> None:
    span = {"status": {"message": "https://me:s3cret@host.example/v1 and http://tok@h/ failed"}}
    stored = records.prepare(span, capture_content=True)
    assert stored["status"]["message"] == "https://***@host.example/v1 and http://***@h/ failed"


def test_a_decision_model_needs_its_url_variable_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENVF_URL", raising=False)
    cfg = ModelConfig(id="j", provider="jev", kind="decision", base_url_env="ENVF_URL")
    registry = Registry({"j": cfg})
    with pytest.raises(ConfigError, match="ENVF_URL"):
        mk.decision("j", registry=registry, sink=mk.records.NullSink())
