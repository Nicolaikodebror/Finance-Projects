"""Load and minimally validate the YAML configuration."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = ("valuation_date", "data", "conventions", "curve")


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config, check required sections and parse the valuation date.

    Relative data paths are resolved against the project root (the parent of
    the ``config/`` folder), so scripts work from any working directory.
    """
    path = Path(path).resolve()
    with path.open() as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise KeyError(f"config {path} is missing sections: {missing}")
    if not isinstance(cfg["valuation_date"], date):
        cfg["valuation_date"] = date.fromisoformat(str(cfg["valuation_date"]))
    root = path.parent.parent
    cfg["project_root"] = root
    cfg["data"] = {k: (root / v) for k, v in cfg["data"].items()}
    return cfg
