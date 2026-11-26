# danish-callables

Pricing and risk for Danish fixed-rate callable mortgage bonds (konverterbare
realkreditobligationer): zero curve → Hull-White 1F → Monte Carlo → prepayment and
delivery option → price, OAS, option-adjusted duration and convexity.

> Work in progress. Stages 1-2 of 6 (scaffold, curve, Hull-White) are done.

```bash
pip install -e ".[dev]"
python scripts/make_synthetic_data.py
pytest
```

The full README, with the Danish market features and key charts, comes in stage 6.
