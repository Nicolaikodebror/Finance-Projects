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
