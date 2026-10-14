"""
M1 smoke test of the legacy FastAPI server (src/server.py) through FastAPI's TestClient.
Checks the contract the Next.js app depends on: GET /health and POST /generate for all
10 topics x 3 difficulties (default generator source), plus the opt-in neural path.

Usage (from mathvis-model/):
    .venv/Scripts/python.exe research/audit_scripts/server_smoke.py
"""

import importlib
import io
import os
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
except Exception:
    pass

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

# Same torch/WMI import workaround the legacy server applies (needed before any torch import).
import platform as _platform
try:
    _uc = _platform.uname_result("Windows", "localhost", "10", "10.0", "AMD64")
    _uc.__dict__["processor"] = ""
    _platform._uname_cache = _uc
except Exception:
    pass
_platform.win32_ver = lambda *a, **k: ("10", "10.0", "", "")
_platform._wmi_query = lambda *a, **k: None

from fastapi.testclient import TestClient  # noqa: E402


def run(source: str, pairs: list[tuple[str, str]]) -> tuple[int, int]:
    os.environ["MATHGEN_SOURCE"] = source
    import server
    server = importlib.reload(server)   # GEN_SOURCE is read at import time
    ok = 0
    with TestClient(server.app) as client:  # context manager triggers the startup model load
        health = client.get("/health").json()
        print(f"[{source}] /health -> {health}")
        for topic, diff in pairs:
            r = client.post("/generate", json={"topic_id": topic, "difficulty": diff})
            body = r.json() if r.status_code == 200 else {}
            valid = (r.status_code == 200 and isinstance(body.get("questionText"), str)
                     and isinstance(body.get("hints"), list) and "answerKey" in body)
            ok += valid
            if not valid:
                print(f"  FAIL {topic}/{diff}: {r.status_code} {r.text[:120]}")
    return ok, len(pairs)


def main() -> None:
    from dataset import TOPIC_GENERATORS
    pairs = [(t, d) for t in TOPIC_GENERATORS for d in ("easy", "medium", "hard")]
    ok, n = run("generator", pairs)
    print(f"[generator] /generate valid responses: {ok}/{n}")
    ok, n = run("model", pairs[:3])
    print(f"[model]     /generate valid responses: {ok}/{n} (verifier-in-the-loop path)")


if __name__ == "__main__":
    main()
