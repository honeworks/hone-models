"""A fake standalone project that speaks hone-models' `command` protocol (tests only).

    python fake_project.py <request.json>

The input `mode` picks what it does: `ok` (a WAV listed in result.json), `unlisted` (two WAVs, no
result.json), `error` (an out-of-memory error in result.json), `empty` (exit 0, no files), `crash` (60
lines on stderr, exit 3), `hang` (a child process, then sleeps) and `stubborn` (like `hang`, but ignores
SIGTERM). It copies the request and its working directory to `$FAKE_SEEN/seen.json`, and for `hang` /
`stubborn` writes its own and its child's pids to `$FAKE_SEEN/pids.json`.
"""

import json
import os
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path


def silence(path: Path, seconds: float = 1.0) -> str:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\0\0" * int(8000 * seconds))
    return path.name


def hang(seen: Path, *, stubborn: bool) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    if stubborn:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # after the child starts: it keeps the default
    (seen / "pids.json").write_text(json.dumps([os.getpid(), child.pid]))
    print("working...", file=sys.stderr, flush=True)
    time.sleep(600)


def main() -> int:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_dir = Path(request["out_dir"])
    mode = request["inputs"].get("mode", "ok")
    seen = Path(os.environ["FAKE_SEEN"])
    env = {k: os.environ.get(k) for k in ("VIRTUAL_ENV", "PYTHONPATH", "FAKE_SEEN")}
    record = {"request": request, "cwd": os.getcwd(), "env": env}
    (seen / "seen.json").write_text(json.dumps(record))
    print(f"fake project: mode {mode}", file=sys.stderr)
    result = out_dir / "result.json"
    if mode == "ok":
        name = silence(out_dir / "take.wav", float(request["inputs"].get("duration_s", 1)))
        result.write_text(json.dumps({"files": [name], "error": None, "meta": {"steps": 3}}))
    elif mode == "unlisted":
        silence(out_dir / "b.wav")
        silence(out_dir / "a.wav")
    elif mode == "error":
        result.write_text(json.dumps({"files": [], "error": "torch.OutOfMemoryError: CUDA out of memory."}))
    elif mode == "crash":
        for i in range(60):
            print(f"stderr line {i}", file=sys.stderr)
        return 3
    elif mode in ("hang", "stubborn"):
        hang(seen, stubborn=mode == "stubborn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
