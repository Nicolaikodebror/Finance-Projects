"""Shared fixtures: config and the synthetic 'true' curve."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dkcallables.config import load_config
from dkcallables.curve.curve import ZeroCurve
from dkcallables.curve.nelson_siegel import NSSParams, nss_curve
from dkcallables.data.synthetic import true_nss_params

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> dict[str, Any]:
    return load_config(ROOT / "config" / "base.yaml")


@pytest.fixture(scope="session")
def true_params(cfg: dict[str, Any]) -> NSSParams:
    return true_nss_params(cfg)


@pytest.fixture(scope="session")
def true_curve(true_params: NSSParams) -> ZeroCurve:
    return nss_curve(true_params)


@pytest.fixture(scope="session")
def synthetic_dir(cfg: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fresh synthetic data written to a temp dir (tests never depend on data/sample)."""
    from dkcallables.data.synthetic import write_synthetic_data

    out = tmp_path_factory.mktemp("synthetic")
    write_synthetic_data(cfg, out)
    return out
