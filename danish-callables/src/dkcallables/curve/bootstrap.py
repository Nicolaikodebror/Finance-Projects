r"""Bootstrap a zero curve from par swap rates or government bond prices.

Every input is turned into a generic :class:`Instrument`: a set of cash flows
:math:`c_j` at times :math:`t_j` with an observed (dirty) price :math:`V`.
Its model price on a curve is

.. math:: \hat V = \sum_j c_j \, P(0, t_j).

The curve has one knot per instrument, at that instrument's maturity, so
:math:`n` instruments give :math:`n` equations in :math:`n` unknown knot zero
rates. The solution reprices every input exactly.

Algorithm
---------
1. **Sequential bootstrap with linear-in-log-DF interpolation.** Sort by
   maturity. With linear interpolation, the price of instrument :math:`k`
   depends only on knots :math:`1..k`, so knot :math:`k` is solved by a 1-D
   root search (``brentq``) given the earlier knots. This is exact.
2. **Global solve for cubic interpolation.** A cubic spline is non-local: a
   new knot moves the curve between earlier knots, so step 1 no longer
   reprices earlier instruments. We therefore solve all :math:`n` equations
   jointly (``scipy.optimize.root``), starting from the step 1 solution.

Swap valuation assumption
-------------------------
A par swap is valued as a par bond on a single curve: the fixed leg plus a
notional exchange at maturity is worth par when the floating leg is worth par.
This ignores the CIBOR/€STR (DESTR) basis and OIS discounting, see
``docs/assumptions.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.optimize import brentq, root

from dkcallables.conventions import add_months, time_from_valuation
from dkcallables.curve.curve import FloatArray, Interpolation, ZeroCurve


@dataclass(frozen=True)
class Instrument:
    """Generic fixed cash-flow instrument used as a bootstrap input.

    Attributes
    ----------
    name: label for error messages.
    times: cash-flow times in years from valuation (ACT/365F), increasing.
    cashflows: cash-flow amounts, same length as ``times``.
    price: observed dirty price in the same units as ``cashflows``.
    """

    name: str
    times: FloatArray
    cashflows: FloatArray
    price: float

    def __post_init__(self) -> None:
        if self.times.shape != self.cashflows.shape or self.times.size == 0:
            raise ValueError(f"{self.name}: times and cashflows must match and be non-empty")
        if np.any(self.times <= 0) or np.any(np.diff(self.times) <= 0):
            raise ValueError(f"{self.name}: cash-flow times must be positive and increasing")

    @property
    def maturity(self) -> float:
        return float(self.times[-1])


def price_instrument(curve: ZeroCurve, instrument: Instrument) -> float:
    """Model price :math:`\\sum_j c_j P(0,t_j)`."""
    return float(np.dot(instrument.cashflows, curve.discount(instrument.times)))


# ---------------------------------------------------------------------------
# Instrument constructors
# ---------------------------------------------------------------------------
def par_swap_instrument(tenor: float, par_rate: float, fixed_frequency: int = 1) -> Instrument:
    """Par swap as a par bond: coupons ``rate/freq`` at ``k/freq``, plus 1 at maturity; price 1.

    Accrual fractions are taken as exactly ``1/freq`` (a 30/360-style fixed
    leg) and payment times as ``k/freq`` years. Both are simplifications.
    """
    n = round(tenor * fixed_frequency)
    if n < 1 or not np.isclose(n / fixed_frequency, tenor):
        raise ValueError(f"tenor {tenor} is not a whole number of fixed periods")
    times = np.arange(1, n + 1, dtype=float) / fixed_frequency
    cashflows = np.full(n, par_rate / fixed_frequency)
    cashflows[-1] += 1.0
    return Instrument(f"swap_{tenor:g}y", times, cashflows, 1.0)


def bullet_bond_schedule(
    valuation_date: date, maturity_date: date, frequency: int = 1
) -> tuple[list[date], date]:
    """Future coupon dates of a bullet bond (rolled back from maturity) and the previous one."""
    if maturity_date <= valuation_date:
        raise ValueError("bond has matured")
    step = 12 // frequency
    dates = [maturity_date]
    while True:
        prev = add_months(dates[-1], -step)
        if prev <= valuation_date:
            break
        dates.append(prev)
    return sorted(dates), prev


def government_bond_instrument(
    name: str,
    valuation_date: date,
    maturity_date: date,
    coupon: float,
    clean_price: float,
    frequency: int = 1,
) -> Instrument:
    """Bullet government bond (per 100 face) as a bootstrap instrument.

    Danish government bonds pay annual coupons with ACT/ACT accrual (to verify).
    Dirty price = clean + accrued, accrued = 100 * coupon/freq * (days since last
    coupon / days in coupon period), i.e. ACT/ACT ICMA.
    """
    coupon_dates, _ = bullet_bond_schedule(valuation_date, maturity_date, frequency)
    accrued = accrued_interest_act_act(valuation_date, maturity_date, coupon, frequency)
    times = np.array([time_from_valuation(valuation_date, d) for d in coupon_dates])
    cashflows = np.full(times.size, 100.0 * coupon / frequency)
    cashflows[-1] += 100.0
    return Instrument(name, times, cashflows, clean_price + accrued)


def accrued_interest_act_act(
    valuation_date: date, maturity_date: date, coupon: float, frequency: int = 1
) -> float:
    """Accrued interest per 100 face (ACT/ACT ICMA) for a bullet bond."""
    coupon_dates, previous = bullet_bond_schedule(valuation_date, maturity_date, frequency)
    frac = (valuation_date - previous).days / (coupon_dates[0] - previous).days
    return 100.0 * coupon / frequency * frac


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap(
    instruments: Sequence[Instrument],
    interpolation: Interpolation = "cubic",
    tol: float = 1e-10,
) -> ZeroCurve:
    """Build a :class:`ZeroCurve` that reprices every instrument (see module docstring).

    Raises ``RuntimeError`` if the maximum absolute pricing error exceeds ``tol``.
    """
    insts = sorted(instruments, key=lambda x: x.maturity)
    pillars = np.array([x.maturity for x in insts])
    if np.any(np.diff(pillars) <= 0):
        raise ValueError("instrument maturities must be distinct")

    # --- Step 1: exact sequential bootstrap with linear interpolation ------
    zeros = np.zeros(pillars.size)
    for k, inst in enumerate(insts):

        def residual(z_k: float, k: int = k, inst: Instrument = inst) -> float:
            trial = zeros.copy()
            trial[k] = z_k
            # Knots after k are placeholders; with linear interpolation they
            # do not affect cash flows at or before pillar k.
            trial[k + 1 :] = z_k
            curve = ZeroCurve(pillars, trial, "linear")
            return price_instrument(curve, inst) - inst.price

        zeros[k] = brentq(residual, -0.20, 0.50, xtol=1e-15, rtol=1e-15, maxiter=200)

    curve = ZeroCurve(pillars, zeros, "linear")

    # --- Step 2: joint solve when interpolation is non-local --------------
    if interpolation == "cubic":

        def residuals(z: FloatArray) -> FloatArray:
            c = ZeroCurve(pillars, z, "cubic")
            # Scale each equation by its price so swaps (per 1) and bonds
            # (per 100) are solved to comparable relative accuracy.
            return np.array([(price_instrument(c, x) - x.price) / x.price for x in insts])

        sol = root(residuals, zeros, method="hybr", options={"xtol": 1e-15})
        curve = ZeroCurve(pillars, sol.x, "cubic")

    max_err = max(abs(price_instrument(curve, x) - x.price) for x in insts)
    if max_err > tol:
        raise RuntimeError(f"bootstrap did not converge: max pricing error {max_err:.2e}")
    return curve


def par_swap_rate(curve: ZeroCurve, tenor: float, fixed_frequency: int = 1) -> float:
    r"""Par swap rate implied by a curve: :math:`S = (1 - P(T)) / \sum_k \delta P(t_k)`."""
    n = round(tenor * fixed_frequency)
    times = np.arange(1, n + 1, dtype=float) / fixed_frequency
    dfs = curve.discount(times)
    return float((1.0 - dfs[-1]) / (np.sum(dfs) / fixed_frequency))
