"""`hone-models` command line: inspect the registry and recorded calls.

hone-models models list [--json]
hone-models models show <id> [--json]
hone-models models check <id> [--json]        smoke call; measured tokens/s saved as speed_tok_s
hone-models calls list [--since 1d] [--model X] [--json]
hone-models calls show <span_id>
hone-models calls stats [--by model|provider|tag] [--json]
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

try:
    import typer
    from rich.console import Console
    from rich.table import Column, Table
except ImportError as exc:  # pragma: no cover - tested in a subprocess
    raise SystemExit("the hone-models CLI needs the 'cli' extra: pip install 'hone-models[cli]'") from exc

from .calls import call_stats, duration_ms, find_calls
from .errors import HoneModelsError
from .records import default_store, read_spans
from .registry import ModelConfig, load, remember_speed
from .text import text

app = typer.Typer(no_args_is_help=True, help="Inspect hone-models registry and call records.")
models_app = typer.Typer(no_args_is_help=True, help="Registered models.")
app.add_typer(models_app, name="models")
calls_app = typer.Typer(no_args_is_help=True, help="Recorded model calls.")
app.add_typer(calls_app, name="calls")
console = Console()

RegistryOption = Annotated[
    list[Path] | None, typer.Option("--registry", help="Extra registry TOML file (repeatable).")
]
JsonOption = Annotated[bool, typer.Option("--json", help="Machine-readable output.")]
DbOption = Annotated[
    Path | None, typer.Option("--db", help="Span store (default: $HONE_HOME/models/spans.db).")
]
SMOKE_PROMPT = [{"role": "user", "content": "Say OK."}]
SinceOption = Annotated[str | None, typer.Option(help="30m, 12h, 1d, 2w or an ISO time.")]


def fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(1)


@contextmanager
def user_errors() -> Generator[None]:
    """Report our errors as `error: ...` with exit code 1 instead of a traceback."""
    try:
        yield
    except HoneModelsError as exc:
        raise fail(str(exc)) from exc


def show(data: Any, as_json: bool) -> None:
    """Print `data` as JSON, or a dict as `key: value` lines."""
    if as_json:
        print(json.dumps(data, indent=2))
        return
    for key, value in data.items():
        console.print(f"[bold]{key}[/bold]: {value}")


def describe(cfg: ModelConfig) -> dict[str, Any]:
    """The `--json` shape of one model: its config plus `name` and `local`."""
    return {**cfg.model_dump(mode="json"), "name": cfg.name, "local": cfg.local}


@models_app.command("list")
def models_list(registry: RegistryOption = None, as_json: JsonOption = False) -> None:
    """List registered models."""
    with user_errors():
        models = sorted(load(registry).models.values(), key=lambda m: m.id)
    if as_json:
        print(json.dumps([describe(m) for m in models], indent=2))
        return
    table = Table("id", "provider", "kind", "model", "local", "vision", "context")
    for m in models:
        caps = m.capabilities
        table.add_row(
            m.id, m.provider, m.kind, m.name, str(m.local), str(caps.vision), str(caps.max_input_tokens)
        )
    console.print(table)


@models_app.command("show")
def models_show(model_id: str, registry: RegistryOption = None, as_json: JsonOption = False) -> None:
    """Show one model (registered id or ad-hoc provider:model)."""
    with user_errors():
        cfg = load(registry).get(model_id)
    show(describe(cfg), as_json)


@models_app.command("check")
def models_check(model_id: str, registry: RegistryOption = None, as_json: JsonOption = False) -> None:
    """Smoke-call a chat model; save the measured tokens/s to the user registry as speed_tok_s."""
    with user_errors():
        reg = load(registry)
        llm = text(model_id, registry=reg)
        start = time.monotonic()
        r = llm.complete(SMOKE_PROMPT, max_tokens=64)
        seconds = time.monotonic() - start
    if r.error:
        raise fail(f"{model_id}: {r.error}")
    tokens = r.usage.get("output_tokens")
    speed = tokens / seconds if tokens and seconds > 0 else None
    result: dict[str, Any] = {"id": llm.config.id, "text": r.text, "seconds": round(seconds, 2)}
    result["speed_tok_s"] = round(speed, 1) if speed else None
    if speed:
        result["saved_to"] = str(remember_speed(llm.config, speed, registered=llm.config.id in reg.models))
    show(result, as_json)


def _spans(db: Path | None) -> list[dict[str, Any]]:
    path = db or default_store()
    if not path.is_file():
        raise fail(f"no span store at {path}; pass --db PATH")
    return read_spans(path)


@calls_app.command("list")
def calls_list(
    since: SinceOption = None,
    model: Annotated[str | None, typer.Option(help="Registry id or provider model name.")] = None,
    db: DbOption = None,
    as_json: JsonOption = False,
) -> None:
    """List recorded model calls, oldest first."""
    with user_errors():
        calls = find_calls(_spans(db), since=since, model=model)
    if as_json:
        print(json.dumps(calls, indent=2))
        return
    span_id = Column("span_id", no_wrap=True, min_width=16)  # never cut: `calls show` takes it
    table = Table("start", span_id, "name", "model", "status", "in", "out", "ms")
    for c in calls:
        a = c["attributes"]
        ms = duration_ms(c)
        table.add_row(
            c["start_time"], c["span_id"], c["name"], str(a.get("hone.models.model_id", "-")),
            c["status"]["code"], str(a.get("gen_ai.usage.input_tokens", "")),
            str(a.get("gen_ai.usage.output_tokens", "")), f"{ms:.0f}" if ms is not None else "",
        )  # fmt: skip
    console.print(table)


@calls_app.command("show")
def calls_show(span_id: str, db: DbOption = None) -> None:
    """Print one recorded span as JSON."""
    span = next((s for s in _spans(db) if s["span_id"] == span_id), None)
    if span is None:
        raise fail(f"no span {span_id!r} in the store")
    print(json.dumps(span, indent=2))


@calls_app.command("stats")
def calls_stats(
    by: Annotated[str, typer.Option(help="model, provider or tag (hone.step).")] = "model",
    since: SinceOption = None,
    db: DbOption = None,
    as_json: JsonOption = False,
) -> None:
    """Calls, errors, tokens, cost and mean duration per model, provider or tag."""
    with user_errors():
        rows = call_stats(find_calls(_spans(db), since=since), by)
    if as_json:
        print(json.dumps(rows, indent=2))
        return
    table = Table(by, "calls", "errors", "in", "out", "cost_usd", "mean_ms")
    columns = ("key", "calls", "errors", "input_tokens", "output_tokens", "cost_usd", "mean_ms")
    for r in rows:
        table.add_row(*(str(r[k]) for k in columns))
    console.print(table)


def main() -> None:
    app()
