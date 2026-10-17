import importlib
from pathlib import Path

import pytest

import mathgen.compat  # noqa: F401  (patches platform before any test imports torch)

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "base.yaml"


@pytest.fixture(scope="session")
def legacy_dataset():
    """The legacy `src/dataset.py` module (imports torch)."""
    return importlib.import_module("dataset")


@pytest.fixture(scope="session")
def config():
    from mathgen.data.config import load_config
    return load_config(BASE_CONFIG)
