# Modelling assumptions and simplifications

Each entry gives the simplification, why it is acceptable here, and what a production model at
a bank would do instead. The list grows with each stage.

## Conventions and data

| # | Simplification | Production practice |
|---|----------------|---------------------|
| C1 | Conventions in `conventions.py` (quarterly terms on 1 Jan/Apr/Jul/Oct, coupon = annual/4, 30E/360 accrual, 2-month notice) are taken from general market knowledge and marked **to verify**. | Read from the issuer's final terms and the Nasdaq Copenhagen rulebook for each series. |
| C2 | Model time is ACT/365F from the valuation date, and payment dates are not business-day adjusted. | Use a Danish holiday calendar and the following-business-day convention for payment dates. |
| C3 | Ex-coupon dates and the settlement lag (T+2) are ignored. Price is valued as of the valuation date. | Danish mortgage bonds go ex-coupon before the term date, and the price reflects settlement date. Handle the ex-coupon period, where the buyer does not receive the next coupon, explicitly. |
| C4 | Government bonds are assumed to pay annual coupons, with ACT/ACT ICMA accrued interest. | Use the security master for each bond's exact schedule. |

## Curve

| # | Simplification | Production practice |
|---|----------------|---------------------|
| K1 | Single-curve set-up: par swaps are valued as par bonds and discounted on the same curve. The CIBOR-DESTR basis and collateral (OIS) discounting are ignored. | Multi-curve: an OIS (DESTR) discount curve plus a CIBOR 6M projection curve, bootstrapped jointly from OIS, FRAs and swaps. |
| K2 | Swap fixed leg: exactly 1/freq accrual per period, payment at k/freq years, no spot lag. | Exact schedule with the spot lag, business-day adjustment, and 30/360 accrual. |
| K3 | Interpolation is a not-a-knot cubic spline on log discount factors. It gives a smooth forward curve, which Hull-White needs, but it can oscillate between widely spaced knots. | Monotone-convex (Hagan-West 2006) or tension splines, which keep forwards positive and local. |
| K4 | Flat-forward extrapolation beyond the last knot (30y). | Ultimate-forward-rate (UFR) style extrapolation, or market quotes out to 50y. |
| K5 | Government bonds are a **proxy** if swap rates are unavailable. They sit below swaps by the swap spread, which is not constant across maturities. | Use the swap curve. |
| K6 | NSS is fitted with box bounds on the parameters to keep it identified on about 10 bonds. Weights are inverse duration (approximate yield errors). | Central-bank-style fits with exact yield errors, filtering of illiquid or special bonds, and possibly smoothing splines. |

## Hull-White model

| # | Simplification | Production practice |
|---|----------------|---------------------|
| H1 | One factor. All rates are perfectly correlated, so the model cannot produce curve twists. This matters because the refinancing decision depends on the long rate, while discounting depends on the whole path. | Two-factor models (G2++, or LMM-style models) calibrated to the swaption cube. Danish banks typically run 2-factor models for callables. |
| H2 | `a` and `sigma` are set by hand in the config (a = 3%, sigma = 80bp). They are not calibrated to option prices. | Calibrate to the risk-neutral prices of a co-terminal swaption strip (the prepayment option is a Bermudan receiver swaption), or to caps/floors. |
| H3 | The optional history-based helper estimates `(a, sigma)` under the real-world measure from an AR(1) fit. sigma is well identified, but `a` is not: its standard error is about sqrt(2a/T), which is larger than `a` itself on 20 years of data. | Use risk-neutral option prices. If history is used at all, use it only for sigma, and as a sanity check. |
| H4 | Gaussian rates: the short rate can go negative (roughly the bottom 5% of paths at 10y). That is plausible for DKK (negative rates 2012-2022), but it also lets long rates go negative. | Shifted-lognormal or SABR-type vol, or a floor at an effective lower bound, if the desk's view needs one. |
| H5 | Constant `a` and `sigma`. | Piecewise-constant `sigma(t)` to fit a whole strip of swaption vols. |
| H6 | The simulation is exact on any grid: (x, ∫x) are drawn jointly from their bivariate normal, so there is no discretisation error in the path discount factors. The only approximation left in pricing is Monte Carlo noise (with antithetic variates). | Same idea; production systems add quasi-random numbers (Sobol) and control variates. |
