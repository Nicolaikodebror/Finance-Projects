"""Zero curve, bootstrap and Nelson-Siegel-Svensson."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from dkcallables.curve.bootstrap import bootstrap, par_swap_instrument, price_instrument
from dkcallables.curve.curve import ZeroCurve
from dkcallables.curve.nelson_siegel import (
    NSSParams,
    fit_nss,
    fit_nss_to_bonds,
    nss_curve,
    nss_forward,
    nss_zero,
)
from dkcallables.data.loaders import (
    DataValidationError,
    government_instruments,
    load_government_bonds,
    load_swap_rates,
    swap_instruments,
)
from dkcallables.data.synthetic import true_nss_params

# ---------------------------------------------------------------------------
# ZeroCurve
# ---------------------------------------------------------------------------
KNOTS = np.array([0.5, 1, 2, 5, 10, 20, 30])
ZEROS = np.array([0.020, 0.022, 0.025, 0.030, 0.033, 0.035, 0.034])


@pytest.mark.parametrize("interp", ["cubic", "linear"])
def test_curve_reproduces_knots_and_anchor(interp: str) -> None:
    c = ZeroCurve(KNOTS, ZEROS, interp)
    np.testing.assert_allclose(c.zero(KNOTS), ZEROS, atol=1e-14)
    assert c.discount(0.0) == pytest.approx(1.0)


def test_flat_curve() -> None:
    c = ZeroCurve(KNOTS, np.full(KNOTS.size, 0.03))
    t = np.linspace(0.0, 40.0, 81)
    np.testing.assert_allclose(c.instantaneous_forward(t), 0.03, atol=1e-12)
    np.testing.assert_allclose(c.discount(t), np.exp(-0.03 * t), rtol=1e-12)
    np.testing.assert_allclose(c.forward_derivative(t), 0.0, atol=1e-10)


@pytest.mark.parametrize("interp", ["cubic", "linear"])
def test_forward_integrates_to_log_discount(interp: str) -> None:
    """-ln P(0,T) = integral_0^T f(0,u) du, including the extrapolated region."""
    c = ZeroCurve(KNOTS, ZEROS, interp)
    for big_t in [0.3, 3.0, 12.0, 25.0, 40.0]:
        integral, _ = quad(c.instantaneous_forward, 0.0, big_t, points=KNOTS, limit=200)
        assert integral == pytest.approx(-np.log(c.discount(big_t)), abs=1e-9)


def test_cubic_forward_is_continuous_and_derivative_consistent() -> None:
    c = ZeroCurve(KNOTS, ZEROS, "cubic")
    eps = 1e-7
    for k in KNOTS:  # continuity across knots, including the extrapolation start
        assert c.instantaneous_forward(k - eps) == pytest.approx(
            c.instantaneous_forward(k + eps), abs=1e-6
        )
    t = np.array([0.7, 3.3, 15.0])
    h = 1e-5
    fd = (c.instantaneous_forward(t + h) - c.instantaneous_forward(t - h)) / (2 * h)
    np.testing.assert_allclose(c.forward_derivative(t), fd, atol=1e-7)


def test_parallel_shift_moves_forward_exactly() -> None:
    c = ZeroCurve(KNOTS, ZEROS)
    s = c.shifted(0.0001)
    t = np.linspace(0.0, 40.0, 161)
    np.testing.assert_allclose(
        s.instantaneous_forward(t) - c.instantaneous_forward(t), 0.0001, atol=1e-12
    )


def test_curve_input_validation() -> None:
    with pytest.raises(ValueError):
        ZeroCurve([1.0, 0.5], [0.02, 0.02])
    with pytest.raises(ValueError):
        ZeroCurve([0.0, 1.0], [0.02, 0.02])
    with pytest.raises(ValueError):
        ZeroCurve(KNOTS, ZEROS).discount(-1.0)


# ---------------------------------------------------------------------------
# Bootstrap  (required test 1: reprices inputs to within 1e-8)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("interp", ["cubic", "linear"])
def test_bootstrap_swaps_reprices_inputs(synthetic_dir: Path, interp: str) -> None:
    insts = swap_instruments(load_swap_rates(synthetic_dir / "swap_rates.csv"))
    curve = bootstrap(insts, interp)
    for inst in insts:
        assert price_instrument(curve, inst) == pytest.approx(inst.price, abs=1e-8)


@pytest.mark.parametrize("interp", ["cubic", "linear"])
def test_bootstrap_government_bonds_reprices_inputs(synthetic_dir: Path, cfg: dict, interp: str):
    vd = cfg["valuation_date"]
    insts = government_instruments(
        load_government_bonds(synthetic_dir / "government_bonds.csv", vd), vd
    )
    curve = bootstrap(insts, interp)
    for inst in insts:
        assert price_instrument(curve, inst) == pytest.approx(inst.price, abs=1e-8)


def test_bootstrap_recovers_true_curve(synthetic_dir: Path, true_curve: ZeroCurve) -> None:
    """Noise-free swaps from the true curve -> bootstrapped zeros within 0.5bp of the truth."""
    curve = bootstrap(swap_instruments(load_swap_rates(synthetic_dir / "swap_rates.csv")))
    t = np.linspace(1.0, 30.0, 59)
    err_bp = np.abs(curve.zero(t) - true_curve.zero(t)) * 1e4
    assert err_bp.max() < 0.5


def test_bootstrap_single_swap_flat_rate() -> None:
    """A 1y annual par swap at 3% gives z(1) = ln(1.03)."""
    curve = bootstrap([par_swap_instrument(1.0, 0.03)], "linear")
    assert curve.zero(1.0) == pytest.approx(np.log(1.03), abs=1e-12)


# ---------------------------------------------------------------------------
# Nelson-Siegel-Svensson
# ---------------------------------------------------------------------------
def test_nss_limits_and_forward(true_params: NSSParams) -> None:
    p = true_params
    assert nss_zero(0.0, p)[0] == pytest.approx(p.beta0 + p.beta1, abs=1e-8)
    assert nss_zero(1e4, p)[0] == pytest.approx(p.beta0, abs=1e-3)
    # f(t) = d/dt [t z(t)]
    t = np.array([0.5, 2.0, 7.0, 25.0])
    h = 1e-5
    fd = ((t + h) * nss_zero(t + h, p) - (t - h) * nss_zero(t - h, p)) / (2 * h)
    np.testing.assert_allclose(nss_forward(t, p), fd, atol=1e-9)


def test_nss_curve_object_matches_analytic_forward(true_curve: ZeroCurve, true_params) -> None:
    t = np.linspace(0.3, 45.0, 100)
    np.testing.assert_allclose(
        true_curve.instantaneous_forward(t), nss_forward(t, true_params), atol=2e-6
    )


def test_fit_nss_recovers_true_parameters(true_params: NSSParams) -> None:
    t = np.array([0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30])
    fit = fit_nss(t, nss_zero(t, true_params))
    assert fit.success
    assert fit.rmse_bp < 1e-3
    np.testing.assert_allclose(fit.params.as_array(), true_params.as_array(), rtol=1e-3, atol=1e-5)


def test_fit_nss_to_government_bonds(synthetic_dir: Path, cfg: dict) -> None:
    """Fit residuals should be of the order of the injected yield noise (1.5bp)."""
    vd = cfg["valuation_date"]
    insts = government_instruments(
        load_government_bonds(synthetic_dir / "government_bonds.csv", vd), vd
    )
    fit = fit_nss_to_bonds(insts)
    assert fit.success
    assert fit.rmse_bp < 3.0 * cfg["synthetic"]["government_yield_noise_bp"]

    # The fitted curve should track the true government curve (swap curve + spread).
    true_gov = nss_curve(true_nss_params(cfg)).shifted(
        cfg["synthetic"]["government_spread_bp"] * 1e-4
    )
    t = np.linspace(1.0, 25.0, 49)
    err_bp = np.abs(nss_curve(fit.params).zero(t) - true_gov.zero(t)) * 1e4
    assert err_bp.max() < 5.0


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def test_loader_rejects_percent_units(tmp_path: Path) -> None:
    f = tmp_path / "swaps.csv"
    f.write_text("tenor_years,par_rate\n1,2.5\n2,2.7\n")
    with pytest.raises(DataValidationError, match="decimals"):
        load_swap_rates(f)


def test_loader_rejects_missing_column(tmp_path: Path) -> None:
    f = tmp_path / "swaps.csv"
    f.write_text("tenor,par_rate\n1,0.025\n")
    with pytest.raises(DataValidationError, match="missing columns"):
        load_swap_rates(f)
