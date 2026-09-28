"""The README and every ```python block in docs/ run against FakeOllama (examples: test_ac20_examples.py)."""

import re
from pathlib import Path

import pytest

import hone_models as mk
from hone_models.testing import FakeOllama

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).parents[2]
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def python_blocks(page: Path) -> str:
    return "\n".join(re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), re.S))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_doc_snippets_run(page: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mk.gpu.GPU, "memory", lambda: (8192, 0))  # no real GPU in the default suite
    code = python_blocks(page)
    with FakeOllama():
        scope: dict = {"__name__": "__main__"}
        exec(compile(code, str(page), "exec"), scope)  # noqa: S102
    result = scope.get("r")
    assert result is None or result.error is None, result.error


def test_every_doc_page_is_linked_and_has_code() -> None:
    index = (ROOT / "docs" / "README.md").read_text()
    for page in PAGES[1:]:
        if page.name != "README.md":
            assert f"({page.name})" in index, f"{page.name} missing from docs/README.md"
    assert sum(bool(python_blocks(p)) for p in PAGES) >= 10


def test_readme_links_are_absolute_and_point_at_files_here() -> None:
    """PyPI renders the README, so its links are absolute GitHub URLs; each must exist in this repo."""
    readme = (ROOT / "README.md").read_text()
    links = re.findall(r"\]\(([^)]+)\)", readme)
    assert links
    prefix = r"https://github\.com/honeworks/hone-models/(?:blob|tree)/main/(.+)"
    for link in links:
        assert link.startswith("https://"), f"relative link in README.md: {link}"
        local = re.fullmatch(prefix, link)
        if local:
            assert (ROOT / local.group(1)).exists(), f"README.md links to a missing path: {link}"
