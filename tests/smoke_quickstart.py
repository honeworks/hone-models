"""Run the README quickstart against the packaged FakeOllama (used by scripts/check.sh in a fresh venv)."""

import os
import re
import tempfile
from pathlib import Path
from typing import Any

from hone_models.testing import FakeOllama

readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
quickstart = readme.split("## Quickstart", 1)[1]
code = re.search(r"```python\n(.*?)```", quickstart, re.S)
assert code, "no python block in the README quickstart"
os.environ["HONE_HOME"] = tempfile.mkdtemp()
with FakeOllama():
    scope: dict[str, Any] = {"__name__": "__main__"}
    exec(compile(code.group(1), "README quickstart", "exec"), scope)  # noqa: S102
assert scope["r"].error is None, scope["r"].error
assert scope["r"].parsed is not None
print("quickstart ok")
