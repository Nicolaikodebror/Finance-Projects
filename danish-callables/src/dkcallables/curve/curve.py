r"""Zero-coupon curve object.

Notation
--------
* :math:`P(0,t)` discount factor, :math:`\ell(t) = \ln P(0,t)` log discount factor.
* Continuously compounded zero rate :math:`z(t) = -\ell(t)/t`.
* Instantaneous forward :math:`f(0,t) = -\ell'(t)`.

Interpolation is done on :math:`\ell(t)` with knots :math:`(0, 0), (t_1, \ell_1),
\ldots, (t_n, \ell_n)`:

* ``"cubic"``: a not-a-knot cubic spline on :math:`\ell`. :math:`\ell` is
  :math:`C^2`, so the forward :math:`f` is :math:`C^1`. Hull-White needs a
  smooth forward (its drift :math:`\theta(t)` uses :math:`\partial f/\partial t`).
* ``"linear"``: linear in :math:`\ell` = piecewise flat forwards. Robust and
  local, but the forward jumps at the knots, so it is unsuitable for
  Hull-White's :math:`\theta(t)`. Used as the exact first pass of the bootstrap.

Beyond the last knot the curve is extrapolated with a flat forward equal to
:math:`f(0, t_n)`, so :math:`\ell` stays continuous and differentiable.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline

Interpolation = Literal["cubic", "linear"]
FloatArray = NDArray[np.float64]


class ZeroCurve:
    """Continuously compounded zero curve with interpolation on log discount factors.

    Parameters
    ----------
    times:
        Knot times in years, strictly increasing and > 0.
    zero_rates:
        Continuously compounded zero rates at the knots (decimals, 0.03 = 3%).
    interpolation:
        ``"cubic"`` or ``"linear"`` (see module docstring).
    """

    def __init__(
        self,
        times: ArrayLike,
        zero_rates: ArrayLike,
        interpolation: Interpolation = "cubic",
    ) -> None:
        t = np.asarray(times, dtype=float)
        z = np.asarray(zero_rates, dtype=float)
        if t.ndim != 1 or t.shape != z.shape or t.size == 0:
            raise ValueError("times and zero_rates must be 1-D arrays of equal, non-zero length")
        if np.any(t <= 0.0) or np.any(np.diff(t) <= 0.0):
            raise ValueError("times must be strictly increasing and positive")
        if not np.all(np.isfinite(z)):
            raise ValueError("zero_rates must be finite")
        if interpolation not in ("cubic", "linear"):
            raise ValueError(f"unknown interpolation {interpolation!r}")

        self.times: FloatArray = t
        self.zero_rates: FloatArray = z
        self.interpolation: Interpolation = interpolation

        # Knots for log P, including the anchor P(0,0) = 1  =>  l(0) = 0.
        self._knot_t = np.concatenate(([0.0], t))
        self._knot_l = np.concatenate(([0.0], -z * t))

        if interpolation == "cubic":
            if self._knot_t.size < 4:
                raise ValueError("cubic interpolation needs at least 3 curve points")
            self._spline: CubicSpline | None = CubicSpline(
                self._knot_t, self._knot_l, bc_type="not-a-knot"
            )
            self._f_last = float(-self._spline(t[-1], 1))
        else:
            self._spline = None
            # Flat forward on the last segment.
            dl = np.diff(self._knot_l) / np.diff(self._knot_t)
            self._f_last = float(-dl[-1])

    # ------------------------------------------------------------------
    # Core: log discount factor and its derivatives
    # ------------------------------------------------------------------
    def _log_discount(self, t: FloatArray, nu: int = 0) -> FloatArray:
        """``nu``-th derivative of l(t) = ln P(0,t), with flat-forward extrapolation."""
        if np.any(t < 0.0):
            raise ValueError("curve is only defined for t >= 0")
        t_last = self.times[-1]
        l_last = self._knot_l[-1]
        inside = t <= t_last
        out = np.empty_like(t)

        if self._spline is not None:
            out[inside] = self._spline(t[inside], nu)
        else:
            out[inside] = self._linear(t[inside], nu)

        # Extrapolation: l(t) = l(T) - f_last (t - T); l' = -f_last; l'' = 0.
        tail = ~inside
        if nu == 0:
            out[tail] = l_last - self._f_last * (t[tail] - t_last)
        elif nu == 1:
            out[tail] = -self._f_last
        else:
            out[tail] = 0.0
        return out

    def _linear(self, t: FloatArray, nu: int) -> FloatArray:
        kt, kl = self._knot_t, self._knot_l
        if nu == 0:
            return np.interp(t, kt, kl)
        if nu == 1:
            slopes = np.diff(kl) / np.diff(kt)
            # Right-continuous: on [t_{i}, t_{i+1}) use segment i; t = t_n uses the last.
            idx = np.clip(np.searchsorted(kt, t, side="right") - 1, 0, slopes.size - 1)
            return slopes[idx]
        return np.zeros_like(t)

    # ------------------------------------------------------------------
    # Public API (all vectorised; scalars in -> scalars out)
    # ------------------------------------------------------------------
    def discount(self, t: ArrayLike) -> FloatArray | float:
        """Discount factor P(0,t) = exp(l(t))."""
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        return _unwrap(np.exp(self._log_discount(ta)), t)

    def zero(self, t: ArrayLike) -> FloatArray | float:
        """Continuously compounded zero rate z(t) = -l(t)/t; z(0) := f(0,0)."""
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty_like(ta)
        small = ta < 1e-10
        out[~small] = -self._log_discount(ta[~small]) / ta[~small]
        if np.any(small):
            out[small] = -self._log_discount(np.zeros(int(small.sum())), 1)
        return _unwrap(out, t)

    def instantaneous_forward(self, t: ArrayLike) -> FloatArray | float:
        """Instantaneous forward f(0,t) = -dl/dt."""
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        return _unwrap(-self._log_discount(ta, 1), t)

    def forward_derivative(self, t: ArrayLike) -> FloatArray | float:
        """df(0,t)/dt = -d^2 l/dt^2 (needed for Hull-White theta(t))."""
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        return _unwrap(-self._log_discount(ta, 2), t)

    def forward_rate(self, t1: ArrayLike, t2: ArrayLike) -> FloatArray | float:
        """Continuously compounded forward rate between t1 < t2: [l(t1) - l(t2)]/(t2 - t1)."""
        a = np.atleast_1d(np.asarray(t1, dtype=float))
        b = np.atleast_1d(np.asarray(t2, dtype=float))
        if np.any(b <= a):
            raise ValueError("forward_rate requires t2 > t1")
        out = (self._log_discount(a) - self._log_discount(b)) / (b - a)
        return _unwrap(out, t1)

    # ------------------------------------------------------------------
    # Scenario curves (used for duration / key-rate durations)
    # ------------------------------------------------------------------
    def shifted(self, shift: float) -> ZeroCurve:
        """Parallel shift of all knot zero rates by ``shift`` (decimal; 0.0001 = 1bp).

        Since l(t) gains the linear term -shift * t, and both interpolators
        reproduce linear functions exactly, the forward curve also moves by
        exactly ``shift`` everywhere.
        """
        return ZeroCurve(self.times, self.zero_rates + shift, self.interpolation)

    def with_zero_rates(self, zero_rates: ArrayLike) -> ZeroCurve:
        """Same knots and interpolation, new zero rates."""
        return ZeroCurve(self.times, zero_rates, self.interpolation)

    def __repr__(self) -> str:
        return (
            f"ZeroCurve(n_knots={self.times.size}, t_max={self.times[-1]:.2f}, "
            f"interpolation={self.interpolation!r})"
        )


def _unwrap(values: FloatArray, like: ArrayLike) -> FloatArray | float:
    """Return a Python float if the input was a scalar, else the array."""
    return float(values[0]) if np.ndim(like) == 0 else values
