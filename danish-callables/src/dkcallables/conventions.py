"""Danish mortgage-bond market conventions.

Everything in this module is a *convention*, not a model. Each item notes its
source status. "To verify" means: I believe this is the market practice, but
it must be checked against the issuer's bond terms (e.g. Nykredit's
"Final Terms" / base prospectus) and Nasdaq Copenhagen rules before relying on
it.

Conventions implemented
-----------------------
* Quarterly terms: payment dates 1 January, 1 April, 1 July, 1 October.
  (To verify: standard for fixed-rate callable annuity bonds.)
* Quarterly coupon = annual coupon / 4. (To verify.)
* Coupon accrual day count is configurable; default 30E/360, under which every
  full quarter accrues exactly 90/360 = 0.25. (To verify: issuers quote
  ACT/ACT for some series; with quarterly terms the difference is tiny.)
* Prepayment at par requires notice before the term date. Default lag is two
  months (e.g. notice by ~1 November for the 1 January term). (To verify:
  the exact deadline is set by the issuer and has changed over time.)
* Time axis for discounting: ACT/365F year fractions from the valuation
  date. This is a modelling choice, not a market convention.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

TERM_MONTHS: tuple[int, ...] = (1, 4, 7, 10)
"""Months of the four quarterly term dates (always the 1st of the month)."""

PAYMENTS_PER_YEAR: int = 4


class DayCount(StrEnum):
    """Supported day-count conventions."""

    THIRTY_E_360 = "30E/360"
    ACT_365F = "ACT/365F"
    ACT_ACT = "ACT/ACT"


# ---------------------------------------------------------------------------
# Date arithmetic
# ---------------------------------------------------------------------------
def add_months(d: date, months: int) -> date:
    """Add calendar months to ``d``, clamping the day to the month's end.

    Example: 31 Jan + 1 month = 28/29 Feb.
    """
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def is_term_date(d: date) -> bool:
    """True if ``d`` is a quarterly term date (1 Jan/Apr/Jul/Oct)."""
    return d.day == 1 and d.month in TERM_MONTHS


def next_term_date(d: date) -> date:
    """First term date strictly after ``d``."""
    for month in TERM_MONTHS:
        candidate = date(d.year, month, 1)
        if candidate > d:
            return candidate
    return date(d.year + 1, 1, 1)


def previous_term_date(d: date) -> date:
    """Last term date on or before ``d``."""
    for month in reversed(TERM_MONTHS):
        candidate = date(d.year, month, 1)
        if candidate <= d:
            return candidate
    return date(d.year - 1, 10, 1)


def term_dates(start: date, end: date) -> list[date]:
    """All term dates ``t`` with ``start < t <= end``, in increasing order."""
    dates: list[date] = []
    current = next_term_date(start)
    while current <= end:
        dates.append(current)
        current = add_months(current, 3)
    return dates


def notice_deadline(term: date, notice_lag_months: int = 2) -> date:
    """Last date a borrower can give notice to prepay at par on ``term``.

    Modelled as ``term - notice_lag_months`` calendar months. (To verify:
    the actual deadline is issuer-specific.)
    """
    if notice_lag_months < 0:
        raise ValueError("notice_lag_months must be non-negative")
    return add_months(term, -notice_lag_months)


# ---------------------------------------------------------------------------
# Day counts
# ---------------------------------------------------------------------------
def _thirty_e_360(start: date, end: date) -> float:
    """30E/360 (Eurobond basis): both day-of-month values are capped at 30.

    YF = [360 (Y2 - Y1) + 30 (M2 - M1) + (D2 - D1)] / 360.
    """
    d1 = min(start.day, 30)
    d2 = min(end.day, 30)
    days = 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)
    return days / 360.0


def _act_act_isda(start: date, end: date) -> float:
    """ACT/ACT ISDA: days in each calendar year divided by that year's length."""
    if end < start:
        return -_act_act_isda(end, start)
    total = 0.0
    current = start
    while current.year < end.year:
        year_end = date(current.year + 1, 1, 1)
        total += (year_end - current).days / (366 if calendar.isleap(current.year) else 365)
        current = year_end
    total += (end - current).days / (366 if calendar.isleap(end.year) else 365)
    return total


def year_fraction(start: date, end: date, convention: DayCount | str = DayCount.ACT_365F) -> float:
    """Year fraction between two dates under the given day count."""
    convention = DayCount(convention)
    if convention is DayCount.ACT_365F:
        return (end - start).days / 365.0
    if convention is DayCount.THIRTY_E_360:
        return _thirty_e_360(start, end)
    return _act_act_isda(start, end)


def time_from_valuation(valuation_date: date, d: date) -> float:
    """Model time (years, ACT/365F) of date ``d`` measured from the valuation date."""
    return year_fraction(valuation_date, d, DayCount.ACT_365F)


# ---------------------------------------------------------------------------
# Coupon conventions
# ---------------------------------------------------------------------------
def quarterly_coupon_rate(annual_coupon: float) -> float:
    """Quarterly coupon rate = annual coupon / 4 (to verify)."""
    return annual_coupon / PAYMENTS_PER_YEAR


def accrual_factor(start: date, end: date, convention: DayCount | str) -> float:
    """Coupon accrual factor for the period [start, end].

    Under 30E/360 a full quarter between two term dates gives exactly 0.25, so
    the coupon equals annual coupon / 4.
    """
    return year_fraction(start, end, convention)


@dataclass(frozen=True)
class MarketConventions:
    """Bundle of the configurable conventions used across the package."""

    day_count: DayCount = DayCount.THIRTY_E_360
    notice_lag_months: int = 2
    payments_per_year: int = PAYMENTS_PER_YEAR

    def __post_init__(self) -> None:
        if self.payments_per_year != PAYMENTS_PER_YEAR:
            raise ValueError("Only quarterly terms (4 payments per year) are supported.")
        if self.notice_lag_months < 0:
            raise ValueError("notice_lag_months must be non-negative")

    @classmethod
    def from_config(cls, cfg: dict) -> MarketConventions:
        """Build from the ``conventions`` section of the YAML config."""
        return cls(
            day_count=DayCount(cfg.get("day_count", DayCount.THIRTY_E_360)),
            notice_lag_months=int(cfg.get("notice_lag_months", 2)),
            payments_per_year=int(cfg.get("payments_per_year", PAYMENTS_PER_YEAR)),
        )
