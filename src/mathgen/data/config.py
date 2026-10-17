"""Loads and fingerprints the data build config (configs/base.yaml)."""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_sha256(path: str | Path) -> str:
    from .build import normalized_sha256
    return normalized_sha256(Path(path).read_bytes())
