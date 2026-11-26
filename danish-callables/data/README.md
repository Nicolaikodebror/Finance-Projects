# Data

The pipeline runs fully offline on synthetic data. `python scripts/make_synthetic_data.py`
regenerates `data/sample/` from the `synthetic` section of `config/base.yaml`. The seed is
fixed, so the output is identical on every run.

To use real data, put your CSVs in `data/raw/` (git-ignored) and point the `data:` paths in
the config at them. Every loader validates its file and fails with a clear message if a
column is missing, a value is empty, or a rate looks like a percentage instead of a decimal.

**Conventions for all files:** rates are decimals (`0.025` = 2.5%), prices are **clean, per
100 face**, dates are ISO `YYYY-MM-DD`. Lines starting with `#` are ignored, so you can note
the source and download date at the top of the file.

## `swap_rates.csv` — DKK par swap curve (primary curve input)

| column        | type  | description                                           |
|---------------|-------|-------------------------------------------------------|
| `tenor_years` | float | swap tenor in years, whole number of fixed periods     |
| `par_rate`    | float | par fixed rate, decimal, annual fixed leg (to verify) |

Where to get it: Bloomberg (`DKSW<n> Curncy` tickers, CBS library terminals), Refinitiv/LSEG
Eikon, or Danmarks Nationalbank's statistics bank (Statbank tables for swap rates / interest
rates, which carry a lag). Danish callables are priced against the **swap** curve in the market.

## `government_bonds.csv` — Danish government bullets (proxy curve)

| column          | type  | description                              |
|-----------------|-------|------------------------------------------|
| `isin`          | str   | identifier                               |
| `coupon`        | float | annual coupon, decimal                   |
| `maturity_date` | date  | final maturity                           |
| `clean_price`   | float | clean price per 100                      |

Assumed annual coupons with ACT/ACT accrual (to verify). Bonds maturing within 30 days are
dropped. Sources: Danmarks Nationalbank (government debt pages publish the list of government
bonds), Nasdaq Copenhagen bond prices. A government curve is only a **proxy** for the swap
curve: it sits below swaps by the swap spread.

## `true_zero_curve.csv` (synthetic only)

The true zero curve the synthetic data was generated from: `t_years, zero_rate` (continuously
compounded). Used only for reference plots and tests.

## Added in later stages

* Callable bond market prices (stage 5).
* Prepayment history `date, isin, coupon, outstanding_before, prepaid_amount, refi_gain_proxy`
  (stage 4).
* Short-rate history for the optional Hull-White parameter helper (stage 2).
