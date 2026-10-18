"""
Import this module before torch.

- Python 3.14 on Windows: `import torch` calls platform.uname(), which can stall in a WMI query
  on this machine (M1 audit; legacy src/server.py has the same workaround). This module
  pre-fills the platform cache.
- torch's deterministic mode (decisions.md I-008) needs a fixed cuBLAS workspace, set before
  cuBLAS initialises.
"""

import os
import platform

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def patch_platform_for_torch() -> None:
    try:
        uname = platform.uname_result("Windows", "localhost", "10", "10.0", "AMD64")
        uname.__dict__["processor"] = ""
        platform._uname_cache = uname
    except Exception:
        pass
    platform.win32_ver = lambda *a, **k: ("10", "10.0", "", "")
    platform._wmi_query = lambda *a, **k: None


if platform.system() == "Windows":
    patch_platform_for_torch()
