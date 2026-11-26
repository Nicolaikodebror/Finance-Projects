"""Generate the synthetic sample data set from the config.

Usage:  python scripts/make_synthetic_data.py [--config config/base.yaml] [--out data/sample]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dkcallables.config import load_config
from dkcallables.data.synthetic import write_synthetic_data

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "config" / "base.yaml", type=Path)
    parser.add_argument("--out", default=ROOT / "data" / "sample", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    for name, path in write_synthetic_data(cfg, args.out).items():
        print(f"wrote {name:<18} -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
