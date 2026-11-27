r"""Hull-White parameters: set by the user, or estimated from a short-rate history.

What proper calibration would do
--------------------------------
A bank calibrates :math:`(a, \sigma)` to the **risk-neutral** prices of
interest-rate options, typically a co-terminal strip of swaptions (e.g. the
"into 30y" diagonal), because the borrower's prepayment option is economically
a Bermudan receiver swaption on the loan. I do not have swaption data, so the
default is to set :math:`(a, \sigma)` in the config. The helper below is a
rough, **historical-measure** sanity check only.

Historical estimator (exact AR(1) form of an OU process)
-------------------------------------------------------
Under a constant long-run level :math:`\mu`, an OU short rate sampled every
:math:`\Delta` years is exactly an AR(1):

.. math::

    r_{k+1} = c + \phi\, r_k + \varepsilon_k, \qquad
    \phi = e^{-a\Delta}, \qquad
    \mathrm{Var}(\varepsilon) = \sigma^2 \frac{1 - e^{-2a\Delta}}{2a}.

OLS gives :math:`\hat\phi`, so :math:`\hat a = -\ln\hat\phi / \Delta` and
:math:`\hat\sigma = \hat s_\varepsilon \sqrt{2\hat a / (1 - \hat\phi^2)}`.

Caveat, and an interview point: :math:`\sigma` is estimated precisely (it is
pinned down by the quadratic variation, so more frequent sampling helps),
but :math:`a` is very poorly identified. Its standard error is roughly
:math:`\sqrt{2a/T}` for a sample of :math:`T` years, so with :math:`a = 0.03`
and 20 years of data the standard error (~0.055) is larger than :math:`a`
itself. OLS is also biased towards faster mean reversion in small samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class HullWhiteParams:
    a: float
    sigma: float
    source: str  # "config" or "history"


@dataclass(frozen=True)
class OUEstimate:
    a: float
    sigma: float
    mu: float
    phi: float
    a_std_error: float  # asymptotic, via the delta method on phi
    n_obs: int


def estimate_ou(short_rates: ArrayLike, dt: float) -> OUEstimate:
    """Estimate (a, sigma, mu) of an OU process from equally spaced data (see module doc)."""
    r = np.asarray(short_rates, dtype=float)
    if r.size < 30:
        raise ValueError("need at least 30 observations")
    y, x = r[1:], r[:-1]
    design = np.column_stack([np.ones_like(x), x])
    (c, phi), *_ = np.linalg.lstsq(design, y, rcond=None)
    if not 0.0 < phi < 1.0:
        raise ValueError(f"AR(1) coefficient {phi:.4f} outside (0,1): no mean reversion detected")
    resid = y - (c + phi * x)
    s2 = float(resid @ resid) / (y.size - 2)
    a = -np.log(phi) / dt
    sigma = np.sqrt(s2 * 2.0 * a / (1.0 - phi**2))
    # se(phi) from OLS; delta method: da/dphi = -1/(phi dt).
    se_phi = np.sqrt(s2 / float(((x - x.mean()) ** 2).sum()))
    return OUEstimate(
        a=float(a),
        sigma=float(sigma),
        mu=float(c / (1.0 - phi)),
        phi=float(phi),
        a_std_error=float(se_phi / (phi * dt)),
        n_obs=int(r.size),
    )


def load_short_rates(path: str | Path) -> pd.DataFrame:
    """Short-rate history CSV with columns ``date, short_rate`` (decimal), sorted by date."""
    df = pd.read_csv(path, comment="#", parse_dates=["date"])
    if not {"date", "short_rate"} <= set(df.columns):
        raise ValueError(f"{path}: expected columns date, short_rate")
    if (df["short_rate"].abs() > 0.5).any():
        raise ValueError(f"{path}: short_rate must be a decimal, not percent")
    return df.sort_values("date").reset_index(drop=True)


def estimate_from_history(df: pd.DataFrame) -> OUEstimate:
    """OU estimate from a history; the sampling interval is the median date spacing."""
    dt = float(df["date"].diff().dt.days.median()) / 365.0
    return estimate_ou(df["short_rate"].to_numpy(), dt)


def hull_white_params(cfg: dict[str, Any]) -> HullWhiteParams:
    """Parameters from the ``hull_white`` config section.

    If ``calibrate_from_history`` is true, a and sigma are taken from the
    short-rate history file in ``data.short_rates`` instead.
    """
    hw = cfg["hull_white"]
    if hw.get("calibrate_from_history", False):
        est = estimate_from_history(load_short_rates(cfg["data"]["short_rates"]))
        return HullWhiteParams(est.a, est.sigma, "history")
    return HullWhiteParams(float(hw["a"]), float(hw["sigma"]), "config")
