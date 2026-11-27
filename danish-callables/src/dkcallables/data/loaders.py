"""Read input CSVs, validate them, and return typed objects.

Schemas are documented in ``data/README.md``. All rates are decimals
(0.025 = 2.5%), prices are clean per 100 face, dates are ISO ``YYYY-MM-DD``.
Validation is deliberately strict: a wrong unit (percent instead of decimal)
is the most common data error, so rates above 50% are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from dkcallables.curve.bootstrap import (
    Instrument,
    government_bond_instrument,
    par_swap_instrument,
)

SWAP_COLUMNS = ("tenor_years", "par_rate")
GOVERNMENT_COLUMNS = ("isin", "coupon", "maturity_date", "clean_price")


class DataValidationError(ValueError):
    """Raised when an input file violates its documented schema."""


@dataclass(frozen=True)
class SwapQuote:
    tenor_years: float
    par_rate: float


@dataclass(frozen=True)
class GovernmentBondQuote:
    isin: str
    coupon: float
    maturity_date: date
    clean_price: float


def _read_csv(path: str | Path, required: tuple[str, ...]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/make_synthetic_data.py` or add real data."
        )
    df = pd.read_csv(path, comment="#")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"{path.name}: missing columns {missing}")
    if df[list(required)].isna().any().any():
        raise DataValidationError(f"{path.name}: empty values in required columns")
    if df.empty:
        raise DataValidationError(f"{path.name}: no rows")
    return df


def _check_rates(values: pd.Series, name: str, file: str) -> None:
    if (values.abs() > 0.5).any():
        raise DataValidationError(
            f"{file}: {name} above 50% - rates must be decimals (0.025), not percent (2.5)"
        )


def load_swap_rates(path: str | Path) -> list[SwapQuote]:
    """Load par swap rates. Columns: ``tenor_years, par_rate``."""
    df = _read_csv(path, SWAP_COLUMNS)
    _check_rates(df["par_rate"], "par_rate", Path(path).name)
    if (df["tenor_years"] <= 0).any() or df["tenor_years"].duplicated().any():
        raise DataValidationError(f"{Path(path).name}: tenors must be positive and unique")
    df = df.sort_values("tenor_years")
    return [SwapQuote(float(r.tenor_years), float(r.par_rate)) for r in df.itertuples()]


def load_government_bonds(path: str | Path, valuation_date: date) -> list[GovernmentBondQuote]:
    """Load government bond quotes. Columns: ``isin, coupon, maturity_date, clean_price``.

    Bonds maturing within 30 days of the valuation date are dropped (their
    prices carry almost no curve information and are dominated by noise).
    """
    df = _read_csv(path, GOVERNMENT_COLUMNS)
    name = Path(path).name
    _check_rates(df["coupon"], "coupon", name)
    if (df["clean_price"] <= 0).any():
        raise DataValidationError(f"{name}: clean_price must be positive")
    df["maturity_date"] = pd.to_datetime(df["maturity_date"]).dt.date
    df = df[[(m - valuation_date).days > 30 for m in df["maturity_date"]]]
    df = df.sort_values("maturity_date")
    return [
        GovernmentBondQuote(str(r.isin), float(r.coupon), r.maturity_date, float(r.clean_price))
        for r in df.itertuples()
    ]


def swap_instruments(quotes: list[SwapQuote], fixed_frequency: int = 1) -> list[Instrument]:
    """Turn swap quotes into bootstrap instruments."""
    return [par_swap_instrument(q.tenor_years, q.par_rate, fixed_frequency) for q in quotes]


def government_instruments(
    quotes: list[GovernmentBondQuote], valuation_date: date, frequency: int = 1
) -> list[Instrument]:
    """Turn government bond quotes into bootstrap / NSS-fit instruments."""
    return [
        government_bond_instrument(
            q.isin, valuation_date, q.maturity_date, q.coupon, q.clean_price, frequency
        )
        for q in quotes
    ]
