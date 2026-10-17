"""Resolve portable project paths and write machine-readable artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Find the checkout independently of the current working directory."""
    return Path(__file__).resolve().parents[2]


def inside_root(root: Path, relative: str) -> Path:
    """Reject configured destinations outside the project directory."""
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path must stay inside the project: {relative}")
    return path


def load_config(path: Path) -> dict[str, Any]:
    """Read YAML without evaluating arbitrary Python objects."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    if config["data"]["competition"] != "hull-tactical-market-prediction":
        raise ValueError("Only the official Hull competition is supported.")
    if config["validation"].get("shuffle") is not False:
        raise ValueError("Time-series data must not be shuffled.")
    return config


def utc_timestamp() -> str:
    """Return an unambiguous machine timestamp, independent of local timezone."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    """Write strict JSON atomically; NaN and Infinity are not valid JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
