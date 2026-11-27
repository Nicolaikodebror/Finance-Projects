r"""Nelson-Siegel-Svensson (NSS) zero curve.

Zero rate (continuously compounded), with :math:`x_i = t/\tau_i`:

.. math::

    z(t) = \beta_0
         + \beta_1 \frac{1 - e^{-x_1}}{x_1}
         + \beta_2 \left[\frac{1 - e^{-x_1}}{x_1} - e^{-x_1}\right]
         + \beta_3 \left[\frac{1 - e^{-x_2}}{x_2} - e^{-x_2}\right]

Instantaneous forward :math:`f(t) = \frac{d}{dt}[t\,z(t)]`:

.. math::

    f(t) = \beta_0 + \beta_1 e^{-x_1} + \beta_2 x_1 e^{-x_1} + \beta_3 x_2 e^{-x_2}

Interpretation: :math:`\beta_0` = long rate, :math:`\beta_0 + \beta_1` = short
rate, :math:`\beta_2, \beta_3` = humps located near :math:`\tau_1, \tau_2`.
Setting :math:`\beta_3 = 0` gives plain Nelson-Siegel. (Svensson 1994,
Nelson & Siegel 1987.)

Fitting
-------
For fixed :math:`(\tau_1, \tau_2)` the model is *linear* in the betas, so
we (1) grid-search the taus, solving the betas by ordinary least squares on
each grid point, then (2) polish all six parameters jointly with
``scipy.optimize.least_squares``. The grid step avoids the well-known local
minima of NSS.

Two objective functions:

* :func:`fit_nss` - residuals are zero-rate errors (inputs are zero yields).
* :func:`fit_nss_to_bonds` - residuals are price errors divided by
  (dirty price x modified duration), i.e. a first-order approximation to yield
  errors. This is how central banks typically fit NSS to coupon bonds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from dkcallables.curve.bootstrap import Instrument
from dkcallables.curve.curve import FloatArray, Interpolation, ZeroCurve


@dataclass(frozen=True)
class NSSParams:
    """Nelson-Siegel-Svensson parameters (rates in decimals, taus in years)."""

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float

    def as_array(self) -> FloatArray:
        return np.array([self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2])

    @classmethod
    def from_array(cls, x: ArrayLike) -> NSSParams:
        b0, b1, b2, b3, t1, t2 = (float(v) for v in np.asarray(x))
        return cls(b0, b1, b2, b3, t1, t2)


def _loadings(t: FloatArray, tau1: float, tau2: float) -> FloatArray:
    """Factor loadings matrix L (len(t) x 4) so that z = L @ beta."""
    t = np.maximum(np.asarray(t, dtype=float), 1e-10)  # z(0) is the t -> 0 limit
    x1, x2 = t / tau1, t / tau2
    g1 = -np.expm1(-x1) / x1  # (1 - e^{-x}) / x, numerically stable
    g2 = -np.expm1(-x2) / x2
    return np.column_stack([np.ones_like(t), g1, g1 - np.exp(-x1), g2 - np.exp(-x2)])


def nss_zero(t: ArrayLike, p: NSSParams) -> FloatArray:
    """NSS zero rate z(t)."""
    betas = np.array([p.beta0, p.beta1, p.beta2, p.beta3])
    return _loadings(np.atleast_1d(np.asarray(t, dtype=float)), p.tau1, p.tau2) @ betas


def nss_forward(t: ArrayLike, p: NSSParams) -> FloatArray:
    """NSS instantaneous forward f(0,t)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    x1, x2 = t / p.tau1, t / p.tau2
    return p.beta0 + p.beta1 * np.exp(-x1) + p.beta2 * x1 * np.exp(-x1) + p.beta3 * x2 * np.exp(-x2)


def nss_curve(
    p: NSSParams,
    grid: ArrayLike | None = None,
    interpolation: Interpolation = "cubic",
) -> ZeroCurve:
    """Sample the NSS curve on a grid and return it as a :class:`ZeroCurve`.

    Default grid: quarterly to 2y, then semi-annual to 50y. With a fine grid
    the cubic spline reproduces the analytic NSS forward closely.
    """
    if grid is None:
        grid = np.concatenate([np.arange(0.25, 2.0, 0.25), np.arange(2.0, 50.0 + 1e-9, 0.5)])
    g = np.asarray(grid, dtype=float)
    return ZeroCurve(g, nss_zero(g, p), interpolation)


@dataclass(frozen=True)
class NSSFit:
    """Result of an NSS fit."""

    params: NSSParams
    residuals_bp: FloatArray  # model - market, in basis points of yield
    rmse_bp: float
    success: bool


_TAU_GRID = np.array([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
# Economically sensible box bounds. Without them NSS is poorly identified on a
# handful of bonds: beta0 can run to -15% while beta3 offsets it, giving a
# curve that fits the bonds but extrapolates nonsensically.
#            beta0  beta1  beta2  beta3  tau1  tau2
_LOWER = np.array([0.0, -0.15, -0.15, -0.15, 0.1, 0.5])
_UPPER = np.array([0.15, 0.15, 0.15, 0.15, 10.0, 30.0])


def _grid_start(t: FloatArray, z: FloatArray, w: FloatArray) -> FloatArray:
    """Best (tau1 < tau2) on a grid, with betas by weighted OLS on zero rates."""
    best, best_x = np.inf, None
    sw = np.sqrt(w)
    for tau1 in _TAU_GRID:
        for tau2 in _TAU_GRID[tau1 < _TAU_GRID]:
            design = _loadings(t, tau1, tau2) * sw[:, None]
            betas, *_ = np.linalg.lstsq(design, z * sw, rcond=None)
            sse = float(np.sum((design @ betas - z * sw) ** 2))
            if sse < best:
                best, best_x = sse, np.concatenate([betas, [tau1, tau2]])
    assert best_x is not None
    return np.clip(best_x, _LOWER + 1e-9, _UPPER - 1e-9)


def fit_nss(
    maturities: ArrayLike,
    zero_rates: ArrayLike,
    weights: ArrayLike | None = None,
) -> NSSFit:
    """Fit NSS by (weighted) least squares on continuously compounded zero rates."""
    t = np.asarray(maturities, dtype=float)
    z = np.asarray(zero_rates, dtype=float)
    w = np.ones_like(t) if weights is None else np.asarray(weights, dtype=float)
    if t.size < 6:
        raise ValueError("NSS has 6 parameters; need at least 6 observations")

    x0 = _grid_start(t, z, w)

    def resid(x: FloatArray) -> FloatArray:
        return np.sqrt(w) * (nss_zero(t, NSSParams.from_array(x)) - z)

    sol = least_squares(resid, x0, bounds=(_LOWER, _UPPER), xtol=1e-15, ftol=1e-15, gtol=1e-15)
    params = NSSParams.from_array(sol.x)
    res_bp = (nss_zero(t, params) - z) * 1e4
    return NSSFit(params, res_bp, float(np.sqrt(np.mean(res_bp**2))), bool(sol.success))


def _price_on_nss(inst: Instrument, p: NSSParams) -> float:
    return float(np.dot(inst.cashflows, np.exp(-nss_zero(inst.times, p) * inst.times)))


def _ytm_cc(inst: Instrument, price: float) -> tuple[float, float]:
    """Continuously compounded yield and its (Macaulay = modified, cc) duration."""
    y = 0.03
    for _ in range(100):  # Newton: price(y) = sum c exp(-y t)
        df = np.exp(-y * inst.times)
        pv = float(np.dot(inst.cashflows, df))
        dpv = -float(np.dot(inst.cashflows * inst.times, df))
        step = (pv - price) / dpv
        y -= step
        if abs(step) < 1e-14:
            break
    df = np.exp(-y * inst.times)
    duration = float(np.dot(inst.cashflows * inst.times, df) / np.dot(inst.cashflows, df))
    return y, duration


def fit_nss_to_bonds(instruments: Sequence[Instrument]) -> NSSFit:
    """Fit NSS to coupon bond prices, minimising approximate yield errors.

    Residual_i = (model price_i - market price_i) / (market price_i * D_i), with
    D_i the (cc) duration at the market yield. Reported residuals are exact
    yield differences (model yield - market yield) in bp.
    """
    insts = list(instruments)
    if len(insts) < 6:
        raise ValueError("NSS has 6 parameters; need at least 6 bonds")
    ytm_dur = [_ytm_cc(x, x.price) for x in insts]
    scale = np.array([x.price * d for x, (_, d) in zip(insts, ytm_dur, strict=True)])
    # Starting point: treat each bond's yield as a zero rate at its duration.
    x0 = _grid_start(
        np.array([d for _, d in ytm_dur]), np.array([y for y, _ in ytm_dur]), np.ones(len(insts))
    )

    def resid(x: FloatArray) -> FloatArray:
        p = NSSParams.from_array(x)
        return np.array([_price_on_nss(inst, p) - inst.price for inst in insts]) / scale

    sol = least_squares(resid, x0, bounds=(_LOWER, _UPPER), xtol=1e-15, ftol=1e-15, gtol=1e-15)
    params = NSSParams.from_array(sol.x)
    model_y = np.array([_ytm_cc(x, _price_on_nss(x, params))[0] for x in insts])
    market_y = np.array([y for y, _ in ytm_dur])
    res_bp = (model_y - market_y) * 1e4
    return NSSFit(params, res_bp, float(np.sqrt(np.mean(res_bp**2))), bool(sol.success))
