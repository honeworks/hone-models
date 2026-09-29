import json
import tomllib
from datetime import UTC, datetime

import pytest
import respx
from typer.testing import CliRunner

import hone_models as mk
from hone_models.calls import call_stats, find_calls, since_cutoff
from hone_models.cli import app
from hone_models.errors import ConfigError
from hone_models.registry import to_toml
from hone_models.testing import FakeOllama

runner = CliRunner()
URL = "http://127.0.0.1:11434"


def reply(content: str, tokens: int = 5) -> dict:
    return {
        "model": "m",
        "message": {"content": content},
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": tokens,
    }


@pytest.fixture
def store(isolated):
    """A default span store with two chats, a failed chat and an emulated decision (decide + chat)."""
    decision = {"ok": {"answer": "yes", "confidence": 0.9, "rationale": "r"}}
    with respx.mock(base_url=URL) as mock:
        route = mock.post("/api/chat")
        route.side_effect = [
            respx.MockResponse(json=reply("a")),
            respx.MockResponse(json=reply("b")),
            respx.MockResponse(json=reply("")),
            respx.MockResponse(json=reply(json.dumps(decision))),
        ]
        mock.post("/api/show").respond(json={})
        mk.text("gemma4-12b").complete([{"role": "user", "content": "1"}])
        mk.text("gemma4-12b").complete([{"role": "user", "content": "2"}], trace={"hone.step": "draft"})
        mk.text("ollama:tiny").complete([{"role": "user", "content": "3"}])
        mk.decision("gemma4-12b").decide("state", {"ok": mk.YesNo("Ok?")})
    return isolated / ".hone" / "models" / "spans.db"


def test_since_cutoff() -> None:
    now = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
    assert since_cutoff("1d", now) == "2026-09-26T12:00:00.000Z"
    assert since_cutoff("30m", now) == "2026-09-27T11:30:00.000Z"
    assert since_cutoff("2026-09-01T00:00:00") == "2026-09-01T00:00:00.000Z"
    with pytest.raises(ConfigError, match="duration like"):
        since_cutoff("yesterday")


def test_find_calls_and_stats(store) -> None:
    spans = mk.records.read_spans(store)
    calls = find_calls(spans)
    assert [c["name"] for c in calls] == ["hone.models.chat"] * 4  # the decide span wraps the last chat
    assert len(find_calls(spans, model="tiny")) == 1
    assert find_calls(spans, since="2999-01-01T00:00:00") == []
    by_model = {r["key"]: r for r in call_stats(calls, "model")}
    assert by_model["gemma4-12b"]["calls"] == 3
    assert by_model["gemma4-12b"]["input_tokens"] == 30
    assert by_model["ollama:tiny"]["errors"] == 1
    assert by_model["ollama:tiny"]["cost_usd"] is None
    assert by_model["ollama:tiny"]["mean_ms"] >= 0
    assert {r["key"]: r["calls"] for r in call_stats(calls, "tag")} == {"-": 3, "draft": 1}
    with pytest.raises(ConfigError, match="--by"):
        call_stats(calls, "colour")


def test_calls_cli(store) -> None:
    listed = runner.invoke(app, ["calls", "list", "--json", "--since", "1d", "--model", "gemma4-12b"])
    assert listed.exit_code == 0, listed.output
    calls = json.loads(listed.output)
    assert len(calls) == 3
    table = runner.invoke(app, ["calls", "list"], env={"COLUMNS": "80"})
    assert table.exit_code == 0
    assert calls[0]["span_id"] in table.output  # never cut: it is what `calls show` takes

    shown = runner.invoke(app, ["calls", "show", calls[0]["span_id"]])
    assert json.loads(shown.output)["span_id"] == calls[0]["span_id"]
    missing = runner.invoke(app, ["calls", "show", "nope"])
    assert missing.exit_code == 1
    assert "no span 'nope'" in missing.output

    stats = runner.invoke(app, ["calls", "stats", "--by", "provider", "--json"])
    assert json.loads(stats.output) == [
        {
            "key": "ollama",
            "calls": 4,
            "errors": 1,
            "input_tokens": 40,
            "output_tokens": 20,
            "cost_usd": None,
            "mean_ms": json.loads(stats.output)[0]["mean_ms"],
        }
    ]
    assert json.loads(stats.output)[0]["mean_ms"] >= 0
    recent = runner.invoke(app, ["calls", "stats", "--since", "2999-01-01", "--json"])
    assert json.loads(recent.output) == []
    by_name = runner.invoke(app, ["calls", "list", "--json", "--model", "tiny"])
    assert len(json.loads(by_name.output)) == 1
    assert runner.invoke(app, ["calls", "stats"]).exit_code == 0
    assert runner.invoke(app, ["calls", "stats", "--by", "colour"]).exit_code == 1
    assert runner.invoke(app, ["calls", "list", "--since", "soon"]).exit_code == 1


def test_calls_cli_without_a_store(isolated) -> None:
    result = runner.invoke(app, ["calls", "list", "--db", str(isolated / "none.db")])
    assert result.exit_code == 1
    assert "no span store" in result.output


def test_models_check_saves_speed(isolated, monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([100.0, 102.0, 102.0, 106.0])  # the smoke call (load included), then the timed reply
    monkeypatch.setattr("hone_models.cli.time.monotonic", lambda: next(ticks))
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").side_effect = [
            respx.MockResponse(json=reply("OK", tokens=50)),
            respx.MockResponse(json=reply("The sea ...", tokens=200)),
        ]
        mock.post("/api/show").respond(json={"capabilities": ["completion"]})
        result = runner.invoke(app, ["models", "check", "ollama:tiny", "--json"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert (out["text"], out["speed_tok_s"], out["output_tokens"], out["seconds"]) == ("OK", 50.0, 200, 4.0)
    assert out["load_s"] == 1.0  # 2 s smoke call less its 50 tokens at 50 tok/s
    user = isolated / "home" / ".config" / "hone" / "models.toml"
    assert out["saved_to"] == str(user)
    entry = tomllib.loads(user.read_text())["models"]["ollama:tiny"]
    assert entry["provider"] == "ollama"
    assert entry["capabilities"]["speed_tok_s"] == 50.0
    assert mk.registry.load().get("ollama:tiny").capabilities.speed_tok_s == 50.0


def test_models_check_times_only_the_warm_reply(isolated, monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([0.0, 30.0, 30.0, 40.0])  # a cold load of 29.9 s must not count in the speed
    monkeypatch.setattr("hone_models.cli.time.monotonic", lambda: next(ticks))
    with FakeOllama() as server:
        server.queue("/api/chat", {"eval_count": 3}, {"eval_count": 200})
        result = runner.invoke(app, ["models", "check", "deepseek-r1-8b", "--json"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert (out["speed_tok_s"], out["load_s"]) == (20.0, 29.85)
    smoke, timed = [r["body"] for r in server.requests if r["path"] == "/api/chat"]
    assert (smoke["think"], timed["think"]) == (False, False)  # thinking = true in the catalog
    assert smoke["options"]["num_predict"] == 1024  # deepseek-r1:8b thinks anyway: room to answer
    assert timed["options"]["num_predict"] == 200
    assert "words" in timed["messages"][0]["content"]


def test_models_check_registered_model_keeps_other_entries(isolated, monkeypatch) -> None:
    user = isolated / "home" / ".config" / "hone" / "models.toml"
    user.parent.mkdir(parents=True)
    user.write_text('[models.mine]\nprovider = "ollama"\nmodel = "x:1b"\ndefaults = { temperature = 0.5 }\n')
    ticks = iter([0.0, 4.0, 4.0, 8.0])
    monkeypatch.setattr("hone_models.cli.time.monotonic", lambda: next(ticks))
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(json=reply("OK", tokens=10))
        result = runner.invoke(app, ["models", "check", "gemma4-12b"])
    assert result.exit_code == 0, result.output
    reg = mk.registry.load()
    assert reg.get("gemma4-12b").capabilities.speed_tok_s == 2.5
    assert reg.get("gemma4-12b").capabilities.max_input_tokens == 32768  # packaged values still merged
    assert reg.get("mine").defaults == {"temperature": 0.5}


def test_models_check_reports_failures() -> None:
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(json=reply(""))
        empty = runner.invoke(app, ["models", "check", "gemma4-12b"])
    assert empty.exit_code == 1
    assert "empty response" in empty.output
    unknown = runner.invoke(app, ["models", "check", "nope"])
    assert unknown.exit_code == 1
    assert "unknown model 'nope'" in unknown.output
    embedding = runner.invoke(app, ["models", "check", "nomic-embed-text"])
    assert embedding.exit_code == 1
    assert "embedding model" in embedding.output


def test_models_check_by_provider_name_and_without_token_counts(isolated, monkeypatch) -> None:
    ticks = iter([0.0, 2.0, 2.0, 4.0, 0.0, 2.0, 2.0, 4.0])
    monkeypatch.setattr("hone_models.cli.time.monotonic", lambda: next(ticks))
    user = isolated / "home" / ".config" / "hone" / "models.toml"
    with respx.mock(base_url=URL) as mock:
        route = mock.post("/api/chat").respond(json=reply("OK", tokens=10))
        result = runner.invoke(app, ["models", "check", "gemma4-12b:latest", "--json"])
        assert json.loads(result.output)["id"] == "gemma4-12b"
        assert tomllib.loads(user.read_text())["models"]["gemma4-12b"] == {
            "capabilities": {"speed_tok_s": 5.0}
        }
        route.respond(json={"model": "m", "message": {"content": "OK"}, "done_reason": "stop"})
        user.unlink()
        result = runner.invoke(app, ["models", "check", "gemma4-12b", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["speed_tok_s"] is None
    assert not user.exists()


def test_user_registry_round_trips_any_text() -> None:
    models = {
        "m": {"provider": "ollama", "license": 'café 😀 "q" \\ \x7f\n', "defaults": {"stop": ["a", "b"]}}
    }
    assert tomllib.loads(to_toml(models)) == {"models": models}
