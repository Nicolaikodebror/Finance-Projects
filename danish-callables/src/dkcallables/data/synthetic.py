"""Generate synthetic market data from known "true" parameters.

The point of synthetic data is that the truth is known, so tests can check
that each estimation step recovers it. Everything is driven by the
``synthetic`` section of the config and a fixed seed.

Generated files (stage 1)
-------------------------
* ``swap_rates.csv``      - par swap rates implied by the true NSS curve (no noise).
* ``government_bonds.csv``- bullet bonds priced off (true curve + spread),
                             with i.i.d. yield noise per bond.
* ``true_zero_curve.csv`` - the true zero curve on a grid, for reference plots.

* ``short_rates.csv``    - weekly short-rate history from an OU process with
                             known (a, sigma, mu), for the calibration helper.

Later stages add the callable bond prices and prepayment history.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dkcallables.curve.bootstrap import (
    accrued_interest_act_act,
    government_bond_instrument,
    par_swap_rate,
)
from dkcallables.curve.curve import ZeroCurve
from dkcallables.curve.nelson_siegel import NSSParams, nss_curve, nss_zero


def true_nss_params(cfg: dict[str, Any]) -> NSSParams:
    """The 'true' NSS parameters from the config's ``synthetic.true_nss`` section."""
    return NSSParams(**{k: float(v) for k, v in cfg["synthetic"]["true_nss"].items()})


def make_swap_rates(
    curve: ZeroCurve, tenors: list[float], fixed_frequency: int = 1
) -> pd.DataFrame:
    """Par swap rates implied by ``curve`` (exact, no noise)."""
    rates = [par_swap_rate(curve, float(t), fixed_frequency) for t in tenors]
    return pd.DataFrame({"tenor_years": [float(t) for t in tenors], "par_rate": rates})


def make_government_bonds(
    curve: ZeroCurve,
    valuation_date: date,
    bonds: list[dict[str, Any]],
    spread_bp: float,
    noise_bp: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Clean prices of bullet government bonds.

    Each bond is discounted at ``z(t) + spread + eps_i`` with
    ``eps_i ~ N(0, noise_bp^2)``: a parallel government-swap spread plus a
    bond-specific yield error (liquidity, specialness).
    """
    rows = []
    for spec in bonds:
        maturity = spec["maturity_date"]
        if not isinstance(maturity, date):
            maturity = date.fromisoformat(str(maturity))
        coupon = float(spec["coupon"])
        # Build the cash flows with a dummy price, then price them.
        inst = government_bond_instrument(spec["isin"], valuation_date, maturity, coupon, 100.0)
        eps = rng.normal(0.0, noise_bp) * 1e-4
        y = np.asarray(curve.zero(inst.times)) + spread_bp * 1e-4 + eps
        dirty = float(np.dot(inst.cashflows, np.exp(-y * inst.times)))
        clean = dirty - accrued_interest_act_act(valuation_date, maturity, coupon)
        rows.append(
            {
                "isin": spec["isin"],
                "coupon": coupon,
                "maturity_date": maturity.isoformat(),
                "clean_price": round(clean, 6),
            }
        )
    return pd.DataFrame(rows)


def make_short_rate_history(
    valuation_date: date,
    years: float,
    a: float,
    sigma: float,
    mu: float,
    r0: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Weekly OU history ending at the valuation date, simulated exactly.

    r_{k+1} = mu + (r_k - mu) e^{-a dt} + sigma sqrt((1 - e^{-2a dt}) / (2a)) Z_k.
    """
    dates = pd.date_range(end=pd.Timestamp(valuation_date), periods=int(years * 52) + 1, freq="7D")
    dt = 7.0 / 365.0
    decay = np.exp(-a * dt)
    sd = sigma * np.sqrt(-np.expm1(-2.0 * a * dt) / (2.0 * a))
    r = np.empty(dates.size)
    r[0] = r0
    for k in range(1, r.size):
        r[k] = mu + (r[k - 1] - mu) * decay + sd * rng.standard_normal()
    return pd.DataFrame({"date": dates.date, "short_rate": r})


def write_synthetic_data(cfg: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Generate all stage-1 synthetic files into ``out_dir`` and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    syn = cfg["synthetic"]
    rng = np.random.default_rng(int(syn["seed"]))
    params = true_nss_params(cfg)
    curve = nss_curve(params)
    freq = int(cfg["curve"].get("swap_fixed_frequency", 1))

    paths = {
        "swap_rates": out / "swap_rates.csv",
        "government_bonds": out / "government_bonds.csv",
        "true_zero_curve": out / "true_zero_curve.csv",
        "short_rates": out / "short_rates.csv",
    }
    make_swap_rates(curve, syn["swap_tenors"], freq).to_csv(paths["swap_rates"], index=False)
    make_government_bonds(
        curve,
        cfg["valuation_date"],
        syn["government_bonds"],
        float(syn["government_spread_bp"]),
        float(syn["government_yield_noise_bp"]),
        rng,
    ).to_csv(paths["government_bonds"], index=False)

    grid = np.arange(0.25, 40.0 + 1e-9, 0.25)
    pd.DataFrame({"t_years": grid, "zero_rate": nss_zero(grid, params)}).to_csv(
        paths["true_zero_curve"], index=False
    )
    h = syn["short_rate_history"]
    make_short_rate_history(
        cfg["valuation_date"],
        float(h["years"]),
        float(h["a"]),
        float(h["sigma"]),
        float(h["mu"]),
        float(h["r0"]),
        rng,
    ).to_csv(paths["short_rates"], index=False, float_format="%.8f")
    return paths
