"""The core must import and work without any optional extra (typer, rich, litellm, pynvml)."""

import subprocess
import sys

CHECK = """
import sys
for name in ("typer", "rich", "litellm", "pynvml"):
    sys.modules[name] = None  # make the extra unimportable
import hone_models
assert not {"typer", "litellm", "pynvml"} & {m for m, v in sys.modules.items() if v is not None}
"""

CLI_WITHOUT_EXTRA = """
import sys
sys.modules["typer"] = None
try:
    import hone_models.cli
except SystemExit as exc:
    assert "hone-models[cli]" in str(exc), exc
else:
    raise AssertionError("expected SystemExit")
"""


def test_core_imports_without_extras() -> None:
    subprocess.run([sys.executable, "-c", CHECK], check=True)


def test_cli_without_extra_says_what_to_install() -> None:
    subprocess.run([sys.executable, "-c", CLI_WITHOUT_EXTRA], check=True)
