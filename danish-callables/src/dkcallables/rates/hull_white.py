r"""Hull-White one-factor short-rate model.

Model (risk-neutral measure)
----------------------------
.. math:: dr(t) = [\theta(t) - a\,r(t)]\,dt + \sigma\,dW(t), \qquad a > 0.

Decomposition (Brigo & Mercurio 2006, sec. 3.3)
-----------------------------------------------
Write :math:`r(t) = x(t) + \alpha(t)` with

.. math::

    dx = -a\,x\,dt + \sigma\,dW,\quad x(0) = 0, \qquad
    \alpha(t) = f(0,t) + \frac{\sigma^2}{2a^2}\left(1 - e^{-at}\right)^2 .

:math:`x` is a zero-mean Ornstein-Uhlenbeck process that does **not** depend on
the initial curve; :math:`\alpha` absorbs the curve. Choosing :math:`\alpha`
this way makes the model reprice the initial curve exactly, which is
equivalent to the closed-form drift

.. math::

    \theta(t) = \frac{\partial f(0,t)}{\partial t} + a f(0,t)
              + \frac{\sigma^2}{2a}\left(1 - e^{-2at}\right).

Analytic zero-coupon bond (affine form)
---------------------------------------
With :math:`B(t,T) = (1 - e^{-a(T-t)})/a`,

.. math::

    P(t,T) = A(t,T)\,e^{-B(t,T)\,r(t)}, \qquad
    A(t,T) = \frac{P(0,T)}{P(0,t)}
             \exp\!\Big(B(t,T) f(0,t) - \frac{\sigma^2}{4a}(1 - e^{-2at}) B(t,T)^2\Big).

Equivalently, in terms of :math:`x(t)`,

.. math::

    P(t,T) = \frac{P(0,T)}{P(0,t)}
             \exp\!\Big(\tfrac12[V(t,T) - V(0,T) + V(0,t)] - B(t,T)\,x(t)\Big),

with :math:`V(t,T) = \mathrm{Var}\big[\int_t^T x\,du \,|\, \mathcal F_t\big]
= \frac{\sigma^2}{a^2}\big[(T-t) - 2B(t,T) + \frac{1 - e^{-2a(T-t)}}{2a}\big]`.

Exact simulation
----------------
Over a step :math:`\Delta`, the pair :math:`(x_{t+\Delta}, I_{t,t+\Delta})`
with :math:`I = \int_t^{t+\Delta} x\,du` is bivariate normal given
:math:`x_t`:

.. math::

    \begin{aligned}
    E[x_{t+\Delta}] &= x_t e^{-a\Delta}, &
    \mathrm{Var}[x_{t+\Delta}] &= \sigma^2 \frac{1 - e^{-2a\Delta}}{2a}, \\
    E[I] &= x_t B(\Delta), &
    \mathrm{Var}[I] &= V(\Delta), \\
    \mathrm{Cov}[x_{t+\Delta}, I] &= \frac{\sigma^2}{2a^2}(1 - e^{-a\Delta})^2 .
    \end{aligned}

Sampling both exactly gives the path discount factor without any
discretisation error:

.. math::

    D(0,t) = e^{-\int_0^t r\,du}
           = P(0,t)\, e^{-\frac12 V(0,t)}\, e^{-\int_0^t x\,du},

and :math:`E[D(0,t)] = P(0,t)` exactly (lognormal mean), so Monte Carlo zero
bond prices match the curve up to pure sampling error.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dkcallables.conventions import term_dates, time_from_valuation
from dkcallables.curve.curve import ZeroCurve

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------
def simulation_grid(valuation_date: date, end_date: date, substeps: int = 1) -> FloatArray:
    """Model times (ACT/365F) of 0 and every term date up to ``end_date``.

    Simulating on the actual term dates means cash-flow dates are grid points.
    ``substeps > 1`` splits each period into equal sub-steps (finer grid). The
    first period is a stub from the valuation date to the next term date.
    """
    if substeps < 1:
        raise ValueError("substeps must be >= 1")
    knots = [0.0] + [
        time_from_valuation(valuation_date, d) for d in term_dates(valuation_date, end_date)
    ]
    if len(knots) == 1:
        raise ValueError("no term dates between valuation and end date")
    grid = [0.0]
    for t0, t1 in zip(knots[:-1], knots[1:], strict=True):
        grid.extend(np.linspace(t0, t1, substeps + 1)[1:])
    return np.asarray(grid)


# ---------------------------------------------------------------------------
# Factor paths (curve-independent part of the model)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FactorPaths:
    """Simulated OU factor.

    Attributes
    ----------
    times: grid (n_steps + 1,), starting at 0.
    x: x(t_k), shape (n_paths, n_steps + 1), x[:, 0] = 0.
    int_x: cumulative integral int_0^{t_k} x du, same shape, int_x[:, 0] = 0.
    antithetic: True if paths come in pairs (row i, row i + n/2) with opposite shocks.
    """

    times: FloatArray
    x: FloatArray
    int_x: FloatArray
    antithetic: bool

    @property
    def n_paths(self) -> int:
        return self.x.shape[0]


def _b(a: float, tau: FloatArray | float) -> FloatArray | float:
    """B(tau) = (1 - exp(-a tau)) / a."""
    return -np.expm1(-a * np.asarray(tau)) / a


def _v(a: float, sigma: float, tau: FloatArray | float) -> FloatArray | float:
    """V(tau) = Var[int_0^tau x du] = sigma^2/a^2 [tau - 2 B(tau) + (1 - e^{-2 a tau})/(2a)]."""
    tau = np.asarray(tau)
    return sigma**2 / a**2 * (tau - 2.0 * _b(a, tau) - np.expm1(-2.0 * a * tau) / (2.0 * a))


def simulate_factor(
    a: float,
    sigma: float,
    times: ArrayLike,
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool = True,
) -> FactorPaths:
    """Exact joint simulation of (x, int x) on an arbitrary increasing grid (see module doc)."""
    _check_params(a, sigma)
    t = np.asarray(times, dtype=float)
    if t[0] != 0.0 or np.any(np.diff(t) <= 0):
        raise ValueError("times must start at 0 and be strictly increasing")
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even with antithetic variates")

    n_steps = t.size - 1
    n_draw = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((2, n_draw, n_steps))
    if antithetic:
        z = np.concatenate([z, -z], axis=1)

    x = np.zeros((n_paths, n_steps + 1))
    int_x = np.zeros((n_paths, n_steps + 1))
    for k, dt in enumerate(np.diff(t)):
        decay = np.exp(-a * dt)
        var_x = sigma**2 * (-np.expm1(-2.0 * a * dt)) / (2.0 * a)
        var_i = float(_v(a, sigma, dt))
        cov = sigma**2 / (2.0 * a**2) * (-np.expm1(-a * dt)) ** 2
        # Cholesky of [[var_x, cov], [cov, var_i]].
        l11 = np.sqrt(var_x)
        l21 = cov / l11
        l22 = np.sqrt(max(var_i - l21**2, 0.0))
        x_prev = x[:, k]
        x[:, k + 1] = x_prev * decay + l11 * z[0, :, k]
        int_x[:, k + 1] = int_x[:, k] + x_prev * _b(a, dt) + l21 * z[0, :, k] + l22 * z[1, :, k]
    return FactorPaths(t, x, int_x, antithetic)


# ---------------------------------------------------------------------------
# The model: factor + curve
# ---------------------------------------------------------------------------
def _check_params(a: float, sigma: float) -> None:
    if a <= 1e-6:
        raise ValueError("mean reversion a must be > 1e-6 (the a -> 0 limit is not implemented)")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0:
        raise ValueError("sigma = 0 is degenerate; use a small positive value instead")


@dataclass(frozen=True)
class HullWhite:
    """Hull-White 1F model fitted to ``curve``.

    ``a`` is the mean-reversion speed (per year) and ``sigma`` the absolute
    (normal) short-rate volatility (decimal per sqrt(year); 0.008 = 80bp).
    """

    curve: ZeroCurve
    a: float
    sigma: float

    def __post_init__(self) -> None:
        _check_params(self.a, self.sigma)

    def with_curve(self, curve: ZeroCurve) -> HullWhite:
        """Same (a, sigma) refitted to a new curve, e.g. a shifted one (theta is refitted)."""
        return HullWhite(curve, self.a, self.sigma)

    # --- deterministic functions -------------------------------------------
    def B(self, t: ArrayLike, T: ArrayLike) -> FloatArray:  # noqa: N802 (maths notation)
        """B(t,T) = (1 - exp(-a (T - t))) / a."""
        return np.asarray(_b(self.a, np.asarray(T) - np.asarray(t)))

    def alpha(self, t: ArrayLike) -> FloatArray:
        """alpha(t) = f(0,t) + sigma^2/(2a^2) (1 - e^{-at})^2, so that r = x + alpha."""
        t = np.asarray(t, dtype=float)
        f = np.asarray(self.curve.instantaneous_forward(t))
        return f + self.sigma**2 / (2.0 * self.a**2) * np.expm1(-self.a * t) ** 2

    def theta(self, t: ArrayLike) -> FloatArray:
        """theta(t) = df/dt + a f + sigma^2/(2a) (1 - e^{-2at})."""
        t = np.asarray(t, dtype=float)
        f = np.asarray(self.curve.instantaneous_forward(t))
        dfdt = np.asarray(self.curve.forward_derivative(t))
        return dfdt + self.a * f + self.sigma**2 / (2.0 * self.a) * (-np.expm1(-2.0 * self.a * t))

    def A(self, t: float, T: ArrayLike) -> FloatArray:  # noqa: N802
        """A(t,T) of the affine bond price P(t,T) = A exp(-B r)."""
        T = np.asarray(T, dtype=float)
        b = self.B(t, T)
        p_ratio = np.asarray(self.curve.discount(T)) / self.curve.discount(t)
        f_t = self.curve.instantaneous_forward(t)
        conv = self.sigma**2 / (4.0 * self.a) * (-np.expm1(-2.0 * self.a * t)) * b**2
        return p_ratio * np.exp(b * f_t - conv)

    # --- analytic bond prices on paths --------------------------------------
    def zcb_price(self, t: float, T: ArrayLike, x_t: ArrayLike) -> FloatArray:
        """P(t,T | x_t) for a state vector x_t (n,) and maturities T (m,) -> (n, m).

        Uses the x-form in the module docstring. For T == t the price is 1.
        """
        T = np.atleast_1d(np.asarray(T, dtype=float))
        x_t = np.atleast_1d(np.asarray(x_t, dtype=float))
        if np.any(t - 1e-12 > T):
            raise ValueError("maturities must be >= t")
        p_ratio = np.asarray(self.curve.discount(T)) / self.curve.discount(t)
        v = _v(self.a, self.sigma, T - t) - _v(self.a, self.sigma, T) + _v(self.a, self.sigma, t)
        log_det = np.log(p_ratio) + 0.5 * v  # deterministic part, shape (m,)
        return np.exp(log_det[None, :] - np.outer(x_t, self.B(t, T)))

    def zcb_price_from_r(self, t: float, T: ArrayLike, r_t: ArrayLike) -> FloatArray:
        """P(t,T | r_t) = A(t,T) exp(-B(t,T) r_t), shape (n, m). Same as :meth:`zcb_price`."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        r_t = np.atleast_1d(np.asarray(r_t, dtype=float))
        return self.A(t, T)[None, :] * np.exp(-np.outer(r_t, self.B(t, T)))

    # --- path quantities -----------------------------------------------------
    def short_rate(self, paths: FactorPaths) -> FloatArray:
        """r(t_k) = x(t_k) + alpha(t_k), shape (n_paths, n_steps + 1)."""
        return paths.x + self.alpha(paths.times)[None, :]

    def path_discount(self, paths: FactorPaths) -> FloatArray:
        """D(0,t_k) = exp(-int_0^{t_k} r du) = P(0,t_k) exp(-V(0,t_k)/2 - int_0^{t_k} x du)."""
        t = paths.times
        det = np.asarray(self.curve.discount(t)) * np.exp(-0.5 * _v(self.a, self.sigma, t))
        return det[None, :] * np.exp(-paths.int_x)

    # --- moments (for tests and intuition) -----------------------------------
    def short_rate_mean(self, t: ArrayLike) -> FloatArray:
        """E[r(t)] under the risk-neutral measure = alpha(t)."""
        return self.alpha(t)

    def short_rate_variance(self, t: ArrayLike) -> FloatArray:
        """Var[r(t)] = sigma^2 (1 - e^{-2at}) / (2a)."""
        t = np.asarray(t, dtype=float)
        return self.sigma**2 * (-np.expm1(-2.0 * self.a * t)) / (2.0 * self.a)


# ---------------------------------------------------------------------------
# Monte Carlo statistics
# ---------------------------------------------------------------------------
def mc_mean_and_se(values: ArrayLike, antithetic: bool) -> tuple[FloatArray, FloatArray]:
    """Mean and standard error over paths (axis 0).

    With antithetic variates, rows i and i + n/2 are dependent, so the
    standard error is computed from the n/2 pair averages, which are i.i.d.
    """
    v = np.asarray(values, dtype=float)
    n = v.shape[0]
    if antithetic:
        half = n // 2
        v = 0.5 * (v[:half] + v[half:])
    return v.mean(axis=0), v.std(axis=0, ddof=1) / np.sqrt(v.shape[0])
