"""The catalog (change 0015 §3b): whether a registry model is installed on this machine, and what would
install it.

    catalog.installed(cfg)          # "yes", "no" or "unknown"
    catalog.install_commands(cfg)   # the shell commands that would fetch it (nothing is run)

Where each answer comes from, by the entry's `install` table:

- `dir_env` (a `command` project): the variable names a folder and `check` succeeds in it; unset: unknown.
- `files` (ComfyUI): every file is under `HONE_COMFYUI_DIR` (else `~/ComfyUI`) `/models/<to>`; without
  that folder, ComfyUI's `GET /object_info` lists them; a server that does not answer: unknown.
- `ollama` (or an Ollama entry): `GET /api/tags` names the model; no answer: unknown.
- `hf` (a Hugging Face repository): its snapshot is in the Hugging Face cache.

A hosted model has nothing to install (`yes`); an entry without `install` is `unknown`.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import httpx

from ._registry_shapes import Install, InstallFile
from .errors import ConfigError, ProviderError
from .providers import ollama
from .providers._comfyui_server import default_url
from .registry import ModelConfig

Status = Literal["yes", "no", "unknown"]
HOSTED = ("openai_compatible", "litellm", "jev")
SERVER_TIMEOUT_S = 5.0
CHECK_TIMEOUT_S = 60.0


def comfyui_dir() -> Path:
    """ComfyUI's folder: `HONE_COMFYUI_DIR`, else `~/ComfyUI`."""
    return Path(os.environ.get("HONE_COMFYUI_DIR") or Path.home() / "ComfyUI").expanduser()


def hf_cache() -> Path:
    """The Hugging Face hub cache: `HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    home = os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface"
    return Path(home).expanduser() / "hub"


class Checker:
    """Answers `installed` for many entries, asking each server once (`models list`)."""

    def __init__(self) -> None:
        self._answers: dict[str, set[str] | None] = {}

    def installed(self, cfg: ModelConfig) -> Status:
        """`yes`, `no` or `unknown` (never `no` by guess: a server that does not answer is unknown)."""
        spec = cfg.install or Install()
        if spec.dir_env:
            return _project(spec)
        if spec.files:
            return self._comfyui(cfg, spec.files)
        if spec.ollama or cfg.provider == "ollama":
            return self._ollama(cfg, spec)
        if spec.hf:
            return _yes(_in_hf_cache(spec.hf))
        return "yes" if not cfg.local and cfg.provider in HOSTED else "unknown"

    def _ollama(self, cfg: ModelConfig, spec: Install) -> Status:
        url = ollama.base_url(cfg if cfg.provider == "ollama" else None)
        tags = self._ask(url, "/api/tags", _tag_names)
        if tags is None:
            return "unknown"
        names = [spec.ollama or "", cfg.name if cfg.provider == "ollama" else ""]
        return _yes(any(_with_tag(name) in tags for name in names if name))

    def _comfyui(self, cfg: ModelConfig, files: list[InstallFile]) -> Status:
        models = comfyui_dir() / "models"
        if models.is_dir():
            return _yes(all(_on_disk(models, f) for f in files))
        if os.environ.get("HONE_COMFYUI_DIR"):
            return "unknown"  # the folder named is not there: no guess, and no server asked
        url = cfg.comfyui_url if cfg.provider == "comfyui" else default_url()
        known = self._ask(url, "/object_info", _strings)
        if known is None or any(f.file is None for f in files):
            return "unknown"  # a whole-repository folder is not listed by ComfyUI
        return _yes(all(Path(cast(str, f.file)).name in known for f in files))

    def _ask(self, url: str, path: str, read: Callable[[Any], set[str]]) -> set[str] | None:
        key = f"{url}{path}"
        if key not in self._answers:
            try:
                response = httpx.get(key, timeout=SERVER_TIMEOUT_S)
                response.raise_for_status()
                self._answers[key] = read(response.json())
            except (httpx.HTTPError, ValueError):
                self._answers[key] = None
        return self._answers[key]


def installed(cfg: ModelConfig) -> Status:
    """Whether `cfg` is installed here: `yes`, `no` or `unknown`."""
    return Checker().installed(cfg)


def require_installed(cfg: ModelConfig) -> None:
    """A `ConfigError` before any job when the model is certainly not installed (`unknown` passes)."""
    if installed(cfg) == "no":
        raise ConfigError(
            f"model {cfg.id!r} is not installed on this machine; `hone-models models install {cfg.id}` "
            "prints the commands that fetch it"
        )


def install_commands(cfg: ModelConfig) -> list[str]:
    """The shell commands that would install `cfg`: an `ollama pull`, `hf download`s into ComfyUI's model
    folders, or a clone and the project's setup steps. Empty for a hosted model or without `install`."""
    spec = cfg.install
    if spec is None:
        return []
    commands = [f"ollama pull {spec.ollama}"] if spec.ollama else []
    for item in spec.files:
        commands += _download(item, comfyui_dir() / "models" / item.to)
    if spec.hf:
        commands.append(f"hf download {spec.hf}")
    if spec.repo:
        folder = _project_dir(spec)
        commands.append(f"git clone {spec.repo} {folder}")
        commands += [f"cd {folder} && {step}" for step in spec.setup]
    return commands


def install_folder(cfg: ModelConfig) -> Path:
    """Where the downloads would go (for the free disk space)."""
    spec = cfg.install or Install()
    if spec.files:
        return comfyui_dir() / "models"
    if spec.repo and spec.dir_env and os.environ.get(spec.dir_env):
        return Path(os.environ[spec.dir_env]).expanduser()
    if spec.ollama:
        return Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models").expanduser()
    return hf_cache() if spec.hf else Path.home()


def free_gb(path: Path) -> float | None:
    """Free space on the disk that holds `path` (or its nearest existing parent), in GB."""
    for folder in (path, *path.parents):
        if folder.exists():
            return round(shutil.disk_usage(folder).free / 1e9, 1)
    return None


def run_install(cfg: ModelConfig, run: Callable[..., Any] | None = None) -> list[str]:
    """Run the install commands one by one (the owner's `models install --run`); stop at the first that
    fails. Returns the commands run."""
    commands = install_commands(cfg)
    if not commands:
        raise ConfigError(f"model {cfg.id!r} has no install commands; see its source: {_source(cfg)}")
    for command in commands:
        try:
            (run or subprocess.run)(command, shell=True, check=True)  # noqa: S604 - the owner's own commands
        except subprocess.CalledProcessError as exc:
            raise ProviderError(f"install of {cfg.id!r}: {command!r} exited {exc.returncode}") from exc
    return commands


def _download(item: InstallFile, target: Path) -> list[str]:
    """`hf download` into the ComfyUI folder; a file in a repository subfolder is then moved up, because
    `hf download` keeps the repository's folders."""
    if item.file is None:
        return [f"hf download {item.repo} --local-dir {shlex.quote(str(target))}"]
    commands = [f"hf download {item.repo} {item.file} --local-dir {shlex.quote(str(target))}"]
    if "/" in item.file:
        moved = shlex.quote(str(target / Path(item.file).name))
        commands.append(f"mv {shlex.quote(str(target / item.file))} {moved}")
    return commands


def _project_dir(spec: Install) -> str:
    value = os.environ.get(spec.dir_env or "")
    return shlex.quote(value) if value else f"${spec.dir_env}"


def _project(spec: Install) -> Status:
    value = os.environ.get(cast(str, spec.dir_env))
    if not value:
        return "unknown"
    folder = Path(value).expanduser()
    if not folder.is_dir():
        return "no"
    if not spec.check:
        return "yes"
    try:
        done = subprocess.run(  # noqa: S603 - the registry's own check command, without a shell
            shlex.split(spec.check), cwd=folder, capture_output=True, timeout=CHECK_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return "no"
    return _yes(done.returncode == 0)


def _on_disk(models: Path, item: InstallFile) -> bool:
    folder = models / item.to
    if item.file is None:
        return folder.is_dir() and any(folder.iterdir())
    return (folder / Path(item.file).name).is_file() or (folder / item.file).is_file()


def _in_hf_cache(repo: str) -> bool:
    snapshots = hf_cache() / f"models--{repo.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any(p.is_file() for p in snapshots.rglob("*"))


def _tag_names(data: Any) -> set[str]:
    rows = cast(list[Any], cast(dict[str, Any], data).get("models", [])) if isinstance(data, dict) else []
    return {str(cast(dict[str, Any], r).get("name")) for r in rows if isinstance(r, dict)}


def _strings(data: Any) -> set[str]:
    """Every string in ComfyUI's `/object_info` (the model file choices are lists of names)."""
    found: set[str] = set()
    stack: list[Any] = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            found.add(item)
        elif isinstance(item, dict):
            stack.extend(cast(dict[str, Any], item).values())
        elif isinstance(item, list):
            stack.extend(cast(list[Any], item))
    return found


def _with_tag(name: str) -> str:
    """Ollama names without a tag mean `:latest`."""
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


def _yes(value: bool) -> Status:
    return "yes" if value else "no"


def _source(cfg: ModelConfig) -> str:
    return (cfg.install.source if cfg.install else None) or "none given"
