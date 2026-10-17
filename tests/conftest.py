import importlib
import platform
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "base.yaml"


def _patch_platform_for_torch() -> None:
    """Same workaround the legacy server uses: on Python 3.14/Windows, `import torch` can
    stall in a WMI query via platform.uname(). Pre-fill the cache before importing torch."""
    try:
        uname = platform.uname_result("Windows", "localhost", "10", "10.0", "AMD64")
        uname.__dict__["processor"] = ""
        platform._uname_cache = uname
    except Exception:
        pass
    platform.win32_ver = lambda *a, **k: ("10", "10.0", "", "")
    platform._wmi_query = lambda *a, **k: None


@pytest.fixture(scope="session")
def legacy_dataset():
    """The legacy `src/dataset.py` module (imports torch)."""
    _patch_platform_for_torch()
    return importlib.import_module("dataset")


@pytest.fixture(scope="session")
def config():
    from mathgen.data.config import load_config
    return load_config(BASE_CONFIG)
