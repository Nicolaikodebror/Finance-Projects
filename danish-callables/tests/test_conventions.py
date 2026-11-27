"""Danish quarterly-term conventions."""

from __future__ import annotations

from datetime import date

import pytest

from dkcallables.conventions import (
    DayCount,
    MarketConventions,
    accrual_factor,
    add_months,
    is_term_date,
    next_term_date,
    notice_deadline,
    previous_term_date,
    quarterly_coupon_rate,
    term_dates,
    year_fraction,
)


def test_term_dates_are_quarterly_first_of_month() -> None:
    dates = term_dates(date(2026, 11, 26), date(2027, 12, 31))
    assert dates == [date(2027, 1, 1), date(2027, 4, 1), date(2027, 7, 1), date(2027, 10, 1)]
    assert all(is_term_date(d) for d in dates)


def test_next_and_previous_term_date() -> None:
    assert next_term_date(date(2026, 11, 26)) == date(2027, 1, 1)
    assert next_term_date(date(2027, 1, 1)) == date(2027, 4, 1)  # strictly after
    assert previous_term_date(date(2026, 11, 26)) == date(2026, 10, 1)
    assert previous_term_date(date(2027, 1, 1)) == date(2027, 1, 1)  # on or before
    assert previous_term_date(date(2027, 2, 1)) == date(2027, 1, 1)


def test_full_quarter_accrues_one_quarter_under_30e360() -> None:
    for start, end in [(date(2027, 1, 1), date(2027, 4, 1)), (date(2027, 10, 1), date(2028, 1, 1))]:
        assert accrual_factor(start, end, DayCount.THIRTY_E_360) == pytest.approx(0.25)
    assert quarterly_coupon_rate(0.04) == pytest.approx(0.01)


def test_day_counts() -> None:
    assert year_fraction(date(2027, 1, 1), date(2028, 1, 1), "ACT/365F") == pytest.approx(1.0)
    assert year_fraction(date(2028, 1, 1), date(2029, 1, 1), "ACT/ACT") == pytest.approx(1.0)
    assert year_fraction(date(2028, 1, 1), date(2029, 1, 1), "ACT/365F") == pytest.approx(366 / 365)
    # 30E/360 caps day 31 at 30.
    assert year_fraction(date(2027, 1, 31), date(2027, 3, 31), "30E/360") == pytest.approx(60 / 360)


def test_add_months_clamps_month_end() -> None:
    assert add_months(date(2027, 1, 31), 1) == date(2027, 2, 28)
    assert add_months(date(2027, 1, 1), -2) == date(2026, 11, 1)


def test_notice_deadline_two_months_before_term() -> None:
    assert notice_deadline(date(2027, 1, 1)) == date(2026, 11, 1)
    assert notice_deadline(date(2027, 1, 1), 1) == date(2026, 12, 1)
    with pytest.raises(ValueError):
        notice_deadline(date(2027, 1, 1), -1)


def test_conventions_from_config(cfg: dict) -> None:
    conv = MarketConventions.from_config(cfg["conventions"])
    assert conv.day_count is DayCount.THIRTY_E_360
    assert conv.notice_lag_months == 2
