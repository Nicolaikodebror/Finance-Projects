"""Hull-White 1F: curve fit, analytic bond prices, exact simulation, calibration helper."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dkcallables.conventions import is_term_date
from dkcallables.curve.curve import ZeroCurve
from dkcallables.data.synthetic import make_short_rate_history
from dkcallables.rates.calibration import (
    estimate_from_history,
    estimate_ou,
    hull_white_params,
    load_short_rates,
)
from dkcallables.rates.hull_white import (
    HullWhite,
    mc_mean_and_se,
    simulate_factor,
    simulation_grid,
)

A, SIGMA = 0.03, 0.008
N_PATHS = 20_000


@pytest.fixture(scope="module")
def hw(true_curve: ZeroCurve) -> HullWhite:
    return HullWhite(true_curve, A, SIGMA)


@pytest.fixture(scope="module")
def paths():
    times = np.linspace(0.0, 30.0, 121)  # quarterly to 30y
    return simulate_factor(A, SIGMA, times, N_PATHS, np.random.default_rng(1))


# ---------------------------------------------------------------------------
# Required test 2: MC zero-coupon bond prices match the curve within 3 SE
# ---------------------------------------------------------------------------
def test_mc_zero_coupon_prices_match_curve(hw: HullWhite, paths) -> None:
    disc = hw.path_discount(paths)
    mean, se = mc_mean_and_se(disc, paths.antithetic)
    for T in [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]:
        k = int(np.argmin(np.abs(paths.times - T)))
        target = hw.curve.discount(paths.times[k])
        assert abs(mean[k] - target) < 3.0 * se[k], f"T={T}: {mean[k]:.6f} vs {target:.6f}"


def test_mc_matches_curve_on_irregular_grid(hw: HullWhite) -> None:
    """Exact simulation is grid-independent: a coarse, uneven grid gives the same answer."""
    times = np.array([0.0, 0.1, 1.7, 4.0, 9.5, 20.0])
    p = simulate_factor(A, SIGMA, times, N_PATHS, np.random.default_rng(2))
    mean, se = mc_mean_and_se(hw.path_discount(p), p.antithetic)
    np.testing.assert_array_less(np.abs(mean - hw.curve.discount(times)), 3.0 * se + 1e-15)


def test_discounted_future_bond_price_is_martingale(hw: HullWhite, paths) -> None:
    """E[D(0,t) P(t,T | x_t)] = P(0,T): the analytic P(t,T) used on paths is arbitrage-free."""
    disc = hw.path_discount(paths)
    for t_idx, T in [(20, 10.0), (40, 25.0), (80, 30.0)]:
        t = paths.times[t_idx]
        values = disc[:, t_idx] * hw.zcb_price(t, T, paths.x[:, t_idx])[:, 0]
        mean, se = mc_mean_and_se(values, paths.antithetic)
        assert abs(mean - hw.curve.discount(T)) < 3.0 * se


# ---------------------------------------------------------------------------
# Analytic formulas
# ---------------------------------------------------------------------------
def test_zcb_price_x_form_equals_affine_r_form(hw: HullWhite) -> None:
    t = 3.7
    T = np.array([3.7, 4.0, 10.0, 33.0])
    x = np.array([-0.02, 0.0, 0.015])
    r = x + hw.alpha(t)
    np.testing.assert_allclose(hw.zcb_price(t, T, x), hw.zcb_price_from_r(t, T, r), rtol=1e-12)
    np.testing.assert_allclose(hw.zcb_price(t, t, x), 1.0, rtol=1e-14)


def test_zcb_at_time_zero_is_curve(hw: HullWhite) -> None:
    T = np.array([0.5, 5.0, 30.0])
    np.testing.assert_allclose(hw.zcb_price(0.0, T, 0.0)[0], hw.curve.discount(T), rtol=1e-13)


def test_zcb_decreasing_in_short_rate(hw: HullWhite) -> None:
    prices = hw.zcb_price(5.0, 15.0, np.linspace(-0.03, 0.03, 7))[:, 0]
    assert np.all(np.diff(prices) < 0)


def test_theta_is_consistent_with_alpha(hw: HullWhite) -> None:
    """r = x + alpha and dx = -a x dt + ...  =>  theta(t) = alpha'(t) + a alpha(t)."""
    t = np.array([0.5, 2.0, 7.0, 15.0, 28.0])
    h = 1e-5
    d_alpha = (hw.alpha(t + h) - hw.alpha(t - h)) / (2 * h)
    np.testing.assert_allclose(hw.theta(t), d_alpha + A * hw.alpha(t), atol=1e-8)


def test_theta_flat_curve_closed_form() -> None:
    """Flat curve r0: theta(t) = a r0 + sigma^2/(2a) (1 - e^{-2at})."""
    flat = ZeroCurve([1.0, 10.0, 30.0], [0.03, 0.03, 0.03])
    m = HullWhite(flat, A, SIGMA)
    t = np.array([0.0, 1.0, 10.0])
    expected = A * 0.03 + SIGMA**2 / (2 * A) * (1 - np.exp(-2 * A * t))
    np.testing.assert_allclose(m.theta(t), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Simulation properties
# ---------------------------------------------------------------------------
def test_short_rate_moments(hw: HullWhite, paths) -> None:
    r = hw.short_rate(paths)
    for k in [4, 40, 120]:
        t = paths.times[k]
        mean, se = mc_mean_and_se(r[:, k], paths.antithetic)
        assert abs(mean - hw.short_rate_mean(t)) < 1e-12  # antithetic: x has mean exactly 0
        assert r[:, k].var(ddof=1) == pytest.approx(hw.short_rate_variance(t), rel=0.03)


def test_antithetic_pairs_mirror(paths) -> None:
    half = paths.n_paths // 2
    np.testing.assert_allclose(paths.x[:half], -paths.x[half:], atol=1e-15)
    np.testing.assert_allclose(paths.int_x[:half], -paths.int_x[half:], atol=1e-15)


def test_simulation_is_reproducible() -> None:
    times = np.linspace(0, 5, 21)
    p1 = simulate_factor(A, SIGMA, times, 100, np.random.default_rng(7))
    p2 = simulate_factor(A, SIGMA, times, 100, np.random.default_rng(7))
    np.testing.assert_array_equal(p1.x, p2.x)


def test_common_random_numbers_under_curve_shift(hw: HullWhite, paths) -> None:
    """Shifting the curve leaves the factor paths untouched: same shocks, refitted alpha.

    A +s parallel shift scales every path discount factor by exactly exp(-s t).
    """
    s = 0.0010
    shifted = hw.with_curve(hw.curve.shifted(s))
    ratio = shifted.path_discount(paths) / hw.path_discount(paths)
    np.testing.assert_allclose(
        ratio, np.broadcast_to(np.exp(-s * paths.times), ratio.shape), rtol=1e-12
    )


def test_simulation_grid_hits_term_dates() -> None:
    vd, end = date(2026, 11, 26), date(2028, 1, 1)
    g = simulation_grid(vd, end)
    assert g[0] == 0.0 and g.size == 6  # 0 + 5 term dates
    assert g[1] == pytest.approx(36 / 365)  # stub to 1 Jan 2027
    assert simulation_grid(vd, end, substeps=3).size == 16
    assert is_term_date(date(2028, 1, 1))


def test_invalid_parameters(true_curve: ZeroCurve) -> None:
    with pytest.raises(ValueError):
        HullWhite(true_curve, 0.0, SIGMA)
    with pytest.raises(ValueError):
        HullWhite(true_curve, A, -0.01)
    with pytest.raises(ValueError):
        simulate_factor(A, SIGMA, [0.0, 1.0], 3, np.random.default_rng(0))  # odd + antithetic


# ---------------------------------------------------------------------------
# Calibration helper
# ---------------------------------------------------------------------------
def test_estimate_ou_recovers_parameters_on_long_sample() -> None:
    """Consistency check on a long sample (400y weekly): both parameters recovered."""
    df = make_short_rate_history(
        date(2026, 1, 1), 400, 0.15, 0.008, 0.02, 0.03, np.random.default_rng(3)
    )
    df["date"] = pd.to_datetime(df["date"])
    est = estimate_from_history(df)
    assert est.sigma == pytest.approx(0.008, rel=0.02)
    assert abs(est.a - 0.15) < 3 * est.a_std_error
    assert est.a_std_error < 0.05


def test_sample_history_sigma_precise_a_imprecise(cfg: dict, synthetic_dir: Path) -> None:
    """20y of data: sigma is tight, a is only known to within a wide standard error."""
    true = cfg["synthetic"]["short_rate_history"]
    est = estimate_from_history(load_short_rates(synthetic_dir / "short_rates.csv"))
    assert est.sigma == pytest.approx(true["sigma"], rel=0.05)
    assert abs(est.a - true["a"]) < 3 * est.a_std_error
    assert est.a_std_error > 0.5 * true["a"]  # the identification problem, made explicit


def test_estimate_ou_rejects_explosive_series() -> None:
    """An explosive series (phi > 1) has no mean reversion: the estimator must refuse it."""
    explosive = 0.01 * 1.002 ** np.arange(500)
    with pytest.raises(ValueError, match="no mean reversion"):
        estimate_ou(explosive, 1 / 52)


def test_params_from_config(cfg: dict) -> None:
    p = hull_white_params(cfg)
    assert (p.a, p.sigma, p.source) == (
        cfg["hull_white"]["a"],
        cfg["hull_white"]["sigma"],
        "config",
    )
