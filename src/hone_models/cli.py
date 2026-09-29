"""`hone-models` command line: inspect the registry and recorded calls.

hone-models models list [--kind K] [--feature F] [--installed | --missing] [--tier N] [--json]
hone-models models show <id> [--json]
hone-models models guide [<id>] [--json] [--stale DAYS]   what a model takes; guides not checked lately
hone-models models install <id> [--run]      print the commands that fetch a model (--run: run them)
hone-models models check <id> [--json]        chat: a smoke call (loads the model), then a timed warm reply
                                              whose tokens/s is saved as speed_tok_s; media: a tiny job,
                                              its output and peak GPU memory
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

from . import _media_check, catalog
from .calls import call_stats, duration_ms, find_calls
from .errors import HoneModelsError
from .guide import build, stale
from .media import image, music, video
from .records import default_store, read_spans
from .registry import GENERATION_KEYS, MEDIA_KINDS, ModelConfig, Registry, load, remember_speed, unmet
from .text import TextClient, TextResult, text

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
SMOKE_TOKENS = 1024  # deepseek-r1:8b thinks even with think=false and answers only with room to
SPEED_PROMPT = [{"role": "user", "content": "Describe the sea at dawn in about 200 words."}]
SPEED_TOKENS = 200
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
    """The `--json` shape of one model: its config plus `name` and `local`; the generation keys (change
    0015) only when the entry sets them."""
    unset = {key for key in GENERATION_KEYS if getattr(cfg, key) is None}
    return {**cfg.model_dump(mode="json", exclude=unset), "name": cfg.name, "local": cfg.local}


@models_app.command("list")
def models_list(  # noqa: PLR0917 - a typer command: one parameter per option
    registry: RegistryOption = None,
    as_json: JsonOption = False,
    kind: Annotated[str | None, typer.Option(help="Only this kind (chat, image, music, ...).")] = None,
    feature: Annotated[
        list[str] | None, typer.Option(help="Only models whose guide declares it (repeatable: all of them).")
    ] = None,
    installed: Annotated[bool, typer.Option("--installed", help="Only models installed here.")] = False,
    missing: Annotated[bool, typer.Option("--missing", help="Only models not installed here.")] = False,
    tier: Annotated[int | None, typer.Option(help="Only catalog tier N (1 test first, 3 ceiling).")] = None,
) -> None:
    """List registered models with whether each is installed here (yes, no or unknown)."""
    with user_errors():
        rows = _listed(load(registry), kind, feature or [], tier)
    rows = [(m, s) for m, s in rows if not (installed and s != "yes") and not (missing and s != "no")]
    if as_json:
        print(json.dumps([{**describe(m), "installed": s} for m, s in rows], indent=2))
        return
    ids = Column("id", no_wrap=True)  # never cut: the other commands take it
    table = Table(
        ids,
        "provider",
        "kind",
        Column("model", overflow="fold"),
        Column("installed", min_width=9),
        "local",
        "vision",
        "context",
    )
    for m, status in rows:
        caps = m.capabilities
        table.add_row(
            m.id,
            m.provider,
            m.kind,
            m.name,
            status,
            str(m.local),
            str(caps.vision),
            str(caps.max_input_tokens),
        )
    console.print(table)


def _listed(
    reg: Registry, kind: str | None, features: list[str], tier: int | None
) -> list[tuple[ModelConfig, str]]:
    """The entries that pass the filters, each with its install status (every server asked once)."""
    wanted = {"features": features} if features else {}
    models = sorted(reg.models.values(), key=lambda m: m.id)
    models = [m for m in models if (kind is None or m.kind == kind) and not unmet(m, wanted)]
    models = [m for m in models if tier is None or (m.install is not None and m.install.tier == tier)]
    checker = catalog.Checker()
    return [(m, checker.installed(m)) for m in models]


@models_app.command("show")
def models_show(model_id: str, registry: RegistryOption = None, as_json: JsonOption = False) -> None:
    """Show one model (registered id or ad-hoc provider:model)."""
    with user_errors():
        cfg = load(registry).get(model_id)
    show(describe(cfg), as_json)


@models_app.command("guide")
def models_guide(
    model_id: Annotated[str | None, typer.Argument(help="Registry id; omit with --stale.")] = None,
    registry: RegistryOption = None,
    as_json: JsonOption = False,
    stale_days: Annotated[
        int | None, typer.Option("--stale", help="List the guides not checked for DAYS days.")
    ] = None,
) -> None:
    """Print what a model can take: its guide, inputs, features, limits, licence and install state."""
    with user_errors():
        reg = load(registry)
        if stale_days is not None:
            rows = [(i, str(c) if c else None) for i, c in stale(reg, stale_days) if model_id in (None, i)]
            if as_json:
                print(json.dumps([{"id": i, "checked": c} for i, c in rows], indent=2))
            for i, c in [] if as_json else rows:
                console.print(f"{i}: {c or 'never checked'}")
            return
        if model_id is None:
            raise fail("give a model id, or --stale DAYS")
        g = build(reg.get(model_id))
    print(json.dumps(g.as_dict(), indent=2) if as_json else g.as_text())


@models_app.command("install")
def models_install(
    model_id: str,
    registry: RegistryOption = None,
    run: Annotated[bool, typer.Option("--run", help="Run the commands (downloads; for the owner).")] = False,
) -> None:
    """Print the commands that would fetch a model, with its size and the free disk; download nothing."""
    with user_errors():
        cfg = load(registry).get(model_id)
        print("\n".join(_install_plan(cfg)))
        if run:
            catalog.run_install(cfg)


def _install_plan(cfg: ModelConfig) -> list[str]:
    """What `models install` prints: status, size and free disk as comments, then the commands."""
    spec, folder = cfg.install, catalog.install_folder(cfg)
    size = spec.size_gb if spec and spec.size_gb is not None else "unknown"
    lines = [
        f"# {cfg.id}: installed {catalog.installed(cfg)}; source {spec.source if spec else None}",
        f"# size {size} GB; free on {folder}: {catalog.free_gb(folder)} GB",
        *([f"# {spec.note}"] if spec and spec.note else []),
    ]
    return lines + (catalog.install_commands(cfg) or ["# nothing to download: see the source"])


@models_app.command("check")
def models_check(
    model_id: str,
    registry: RegistryOption = None,
    as_json: JsonOption = False,
    out: Annotated[
        Path | None, typer.Option(help="Media: folder for the tiny output ($HONE_HOME/models/checks).")
    ] = None,
) -> None:
    """Smoke-call a model. Chat: save the measured tokens/s to the user registry as speed_tok_s. Image,
    music, video: run a tiny job in a session and report the output and the peak GPU memory."""
    with user_errors():
        reg = load(registry)
        kind = reg.get(model_id).kind
        if kind in MEDIA_KINDS:
            client = {"image": image, "music": music, "video": video}[kind](model_id, registry=reg)
            found = _media_check.check(client, out or _media_check.default_out_dir())
            if found["error"]:
                raise fail(f"{model_id}: {found['error']}")
            show(found, as_json)
            return
        llm = text(model_id, registry=reg)
        result = _chat_check(llm)
    if speed := result["speed_tok_s"]:
        result["saved_to"] = str(remember_speed(llm.config, speed, registered=llm.config.id in reg.models))
    show(result, as_json)


def _timed(llm: TextClient, prompt: list[dict[str, Any]], max_tokens: int) -> tuple[TextResult, float]:
    start = time.monotonic()
    r = llm.complete(prompt, max_tokens=max_tokens)
    return r, time.monotonic() - start


def _chat_check(llm: TextClient) -> dict[str, Any]:
    """A smoke call that loads the model, then a ~200-token reply timed warm: the speed leaves the load
    out; `load_s` is the smoke call less its own tokens at that speed."""
    smoke, first_s = _timed(llm, SMOKE_PROMPT, SMOKE_TOKENS)
    if smoke.error:
        raise fail(f"{llm.config.id}: {smoke.error}")
    timed, seconds = _timed(llm, SPEED_PROMPT, SPEED_TOKENS)  # cut at the limit or thinking-only: still timed
    tokens = timed.usage.get("output_tokens")
    speed = tokens / seconds if tokens and seconds > 0 else None
    smoke_tokens = smoke.usage.get("output_tokens")
    load = max(first_s - smoke_tokens / speed, 0.0) if speed and smoke_tokens else None
    return {
        "id": llm.config.id,
        "text": smoke.text,
        "load_s": round(load, 2) if load is not None else None,
        "seconds": round(seconds, 2),
        "output_tokens": tokens,
        "speed_tok_s": round(speed, 1) if speed else None,
    }


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
