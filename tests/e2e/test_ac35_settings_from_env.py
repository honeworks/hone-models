"""AC-35: an entry's base URL and key from the environment or a `.env` file, never recorded (change 0017)."""

import os
from pathlib import Path

import pytest
import respx

import hone_models as mk

pytestmark = pytest.mark.e2e
GATEWAY = "https://gateway.example/v1"
KEY = "gw-PLANTED-key-0123456789"
PASSWORD = "hunter2-PLANTED-pass"
REPLY = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "model": "fable"}
JUDGE = """[models.judge]
provider = "openai_compatible"
model = "fable"
base_url_env = "JUDGES_BASE_URL"
api_key_env = "JUDGES_API_KEY"
"""
NAMES = ("JUDGES_BASE_URL", "JUDGES_API_KEY", "AC35_QUOTED", "AC35_SINGLE", "AC35_KEPT", "AC35_FROM_FILE")


@pytest.fixture
def env(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No judge variables set; any a `.env` file sets are removed after the test."""
    for name in NAMES:
        monkeypatch.setenv(name, "")  # registers a restore, so a value set later is removed
        monkeypatch.delenv(name)
    return isolated


def registry(folder: Path, text: str = JUDGE) -> mk.registry.Registry:
    (folder / "hone-models.toml").write_text(text)
    return mk.registry.load()


def ask(reg: mk.registry.Registry, sink: mk.records.SqliteSpanSink | None = None) -> str:
    llm = mk.text("judge", registry=reg, sink=sink if sink is not None else mk.records.NullSink())
    return llm.complete([{"role": "user", "content": "hi"}]).text


def test_ac35_base_url_from_the_environment(env, monkeypatch) -> None:
    monkeypatch.setenv("JUDGES_BASE_URL", GATEWAY)
    monkeypatch.setenv("JUDGES_API_KEY", KEY)
    reg = registry(env)
    assert reg.get("judge").local is False
    with respx.mock(base_url=GATEWAY) as mock:
        route = mock.post("/chat/completions").respond(json=REPLY)
        assert ask(reg) == "ok"
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {KEY}"


def test_ac35_the_variable_beats_base_url(env, monkeypatch) -> None:
    text = JUDGE + 'base_url = "https://fallback.example/v1"\n'
    with respx.mock(base_url="https://fallback.example/v1") as mock:
        mock.post("/chat/completions").respond(json=REPLY)
        monkeypatch.setenv("JUDGES_API_KEY", KEY)
        assert ask(registry(env, text)) == "ok"  # unset: base_url
    monkeypatch.setenv("JUDGES_BASE_URL", GATEWAY)
    with respx.mock(base_url=GATEWAY) as mock:
        route = mock.post("/chat/completions").respond(json=REPLY)
        assert ask(registry(env, text)) == "ok"
    assert route.called


def test_ac35_unset_and_no_base_url_fails_before_any_request(env, monkeypatch) -> None:
    monkeypatch.setenv("JUDGES_API_KEY", KEY)
    reg = registry(env)
    with respx.mock(assert_all_mocked=True) as mock:
        with pytest.raises(mk.errors.ConfigError, match="JUDGES_BASE_URL"):
            ask(reg)
        assert mock.calls.call_count == 0


def test_ac35_dot_env_is_read_and_the_environment_wins(env, monkeypatch) -> None:
    monkeypatch.setenv("AC35_KEPT", "from the shell")
    (env / ".env").write_text(
        "# judges behind the gateway\n"
        "\n"
        f"export JUDGES_BASE_URL={GATEWAY}  # the gateway\n"
        f'JUDGES_API_KEY="{KEY}"\n'
        'AC35_QUOTED="a # not a comment"\n'
        "AC35_SINGLE='single quoted'\n"
        "AC35_KEPT=from the file\n"
    )
    reg = registry(env)
    assert os.environ["JUDGES_BASE_URL"] == GATEWAY
    assert os.environ["AC35_QUOTED"] == "a # not a comment"
    assert os.environ["AC35_SINGLE"] == "single quoted"
    assert os.environ["AC35_KEPT"] == "from the shell"
    with respx.mock(base_url=GATEWAY) as mock:
        route = mock.post("/chat/completions").respond(json=REPLY)
        assert ask(reg) == "ok"
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {KEY}"


def test_ac35_hone_env_file_is_read_before_dot_env(env, monkeypatch) -> None:
    named = env / "judges.env"
    named.write_text(f"JUDGES_BASE_URL={GATEWAY}\nAC35_FROM_FILE=named\n")
    (env / ".env").write_text("JUDGES_BASE_URL=https://other.example/v1\nAC35_KEPT=dot\n")
    monkeypatch.setenv("HONE_ENV_FILE", str(named))
    reg = registry(env)
    assert reg.get("judge").base_url == GATEWAY
    assert (os.environ["AC35_FROM_FILE"], os.environ["AC35_KEPT"]) == ("named", "dot")
    monkeypatch.setenv("HONE_ENV_FILE", str(env / "missing.env"))
    with pytest.raises(mk.errors.ConfigError, match=r"missing\.env"):
        mk.registry.load()


def test_ac35_a_malformed_line_names_file_and_line_not_the_value(env) -> None:
    (env / ".env").write_text(f"JUDGES_BASE_URL={GATEWAY}\n# fine\n{KEY}\n")
    with pytest.raises(mk.errors.ConfigError, match=r"\.env line 3") as caught:
        mk.registry.load()
    assert KEY not in str(caught.value)


def test_ac35_key_and_url_password_never_reach_the_store(env, no_backoff) -> None:
    url = f"https://judge:{PASSWORD}@gateway.example/v1"
    (env / ".env").write_text(f"JUDGES_BASE_URL={url}\nJUDGES_API_KEY={KEY}\n")
    reg = registry(env)
    sink = mk.records.SqliteSpanSink(env / "spans.db")
    with respx.mock(base_url=GATEWAY) as mock:
        mock.post("/chat/completions").respond(json=REPLY)
        assert ask(reg, sink) == "ok"
        mock.post("/chat/completions").respond(400, text="bad request")
        with pytest.raises(mk.errors.ProviderError, match=PASSWORD):  # the caller sees the real URL
            ask(reg, sink)
    sink.close()
    for f in env.glob("spans.db*"):
        data = f.read_bytes()
        assert KEY.encode() not in data
        assert PASSWORD.encode() not in data
    spans = mk.records.read_spans(env / "spans.db")
    assert "https://***@gateway.example/v1" in spans[1]["status"]["message"]
