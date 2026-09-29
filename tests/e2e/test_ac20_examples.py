"""AC-20: every `examples/*.py` runs offline, opens with the What / How / Why docstring, uses only the
public API, asserts what it shows and is listed in `examples/README.md`."""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import hone_models as mk

pytestmark = pytest.mark.e2e
EXAMPLES_DIR = Path(__file__).parents[2] / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("*.py"))
FAKE_NVML = Path(__file__).parents[1] / "fixtures" / "fake_nvml"
PUBLIC_MODULES = {"hone_models", "hone_models.testing", "hone_models.calls"}
DESIGN = Path(__file__).parents[2] / "design" / "current.md"
REQUIRED_EXAMPLES = {  # one per core concept (design/current.md §12)
    "chat_and_structured.py", "registry.py", "prompt_sections_and_replay.py", "decisions.py", "embeddings.py",
    "context_budget.py", "retries_and_errors.py", "gpu_lease.py", "sessions.py", "records.py",
}  # fmt: skip


def tree(script: Path) -> ast.Module:
    return ast.parse(script.read_text(encoding="utf-8"))


def test_ac20_every_required_example_exists() -> None:
    assert REQUIRED_EXAMPLES - {p.name for p in EXAMPLES} == set()


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_ac20_example_runs(script: Path, tmp_path: Path) -> None:
    """Run as a user would, from an empty folder, with a minimal environment: no GPU, no user registry,
    no API keys, no proxies, asserts on. A real local Ollama must never be reached (this runs outside the
    GPU lock), so OLLAMA_HOST points at a closed port and each example starts its own FakeOllama."""
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path / "home"),
        "HONE_HOME": str(tmp_path / ".hone"),
        "PYTHONPATH": str(FAKE_NVML),
        "OLLAMA_HOST": "http://127.0.0.1:9",
        "HONE_COMFYUI_URL": "http://127.0.0.1:9",  # nor a real ComfyUI (installed checks ask /object_info)
    }
    done = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip(), "an example prints what it shows"


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_ac20_example_docstring_explains_what_how_why(script: Path) -> None:
    doc = ast.get_docstring(tree(script)) or ""
    positions = [doc.find(f"\n{head}: ") for head in ("What", "How", "Why")]
    assert -1 not in positions, f"{script.name}: docstring needs 'What:', 'How:' and 'Why:' paragraphs"
    assert positions == sorted(positions), f"{script.name}: What, then How, then Why"


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_ac20_example_asserts_what_it_shows(script: Path) -> None:
    assert any(isinstance(node, ast.Assert) for node in ast.walk(tree(script)))


def private_uses(module: ast.Module) -> list[str]:
    """Imports outside the public modules, underscore names, and `mk.<name>` not in `__all__`."""
    found: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("hone_models"):
            if node.module not in PUBLIC_MODULES or any(a.name.startswith("_") for a in node.names):
                found.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.startswith("hone_models.")]
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") and not node.attr.startswith("__"):
                found.append(f"._{node.attr}")
            elif isinstance(node.value, ast.Name) and node.value.id == "mk" and node.attr not in mk.__all__:
                found.append(f"mk.{node.attr}")
    return found


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_ac20_example_uses_only_the_public_api(script: Path) -> None:
    assert private_uses(tree(script)) == []


def test_private_use_detection() -> None:
    code = "\n".join(
        [
            "from hone_models import _http",
            "from hone_models.text import cost_usd",
            "import hone_models.records",
            "r._SECRETS",
            "mk.budget",
        ]
    )
    assert len(private_uses(ast.parse(code))) == 5


def test_ac20_readme_indexes_every_example_quickstart_first() -> None:
    index = (EXAMPLES_DIR / "README.md").read_text(encoding="utf-8")
    sections = set(re.findall(r"^## (\d+)\. ", DESIGN.read_text(encoding="utf-8"), re.M))
    rows = re.findall(r"^\| \[([\w.]+\.py)\]\(\1\) \|(.*)\|$", index, re.M)  # one table row per example
    assert sorted(name for name, _ in rows) == [p.name for p in EXAMPLES], "one README row per example"
    assert rows[0][0] == "quickstart.py"
    for name, cells in rows:
        concept, sentence, spec = (c.strip() for c in cells.split(" | "))
        assert concept, name
        assert sentence.endswith("."), name
        assert re.fullmatch(r"§\d+(, §\d+)*", spec), f"{name}: Design column {spec!r}"
        for number in re.findall(r"§(\d+)", spec):
            assert number in sections, f"{name}: design/current.md has no section {number}"
