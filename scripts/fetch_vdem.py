"""
Download V-Dem dataset and build data/vdem_snapshot.json.

Requires: pip install vdemdata pandas
Run once: python scripts/fetch_vdem.py

V-Dem dataset is free for academic/research use.
Citation: Coppedge et al. 2024. "V-Dem Dataset v14."
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("fetch_vdem")

_OUT = Path(__file__).parent.parent / "data" / "vdem_snapshot.json"

# Columns to extract
_COLS = [
    "country_name", "year",
    "v2x_polyarchy",   # electoral democracy
    "v2x_libdem",      # liberal democracy
    "v2x_execorrup",   # executive corruption (0=corrupt,1=clean) — actually v2x_execorrup is inverted
    "v2juncind",       # judicial independence
    "v2lgello",        # effective legislative parties (low chambers)
]

# Map V-Dem column to actual direction (True = already 0=bad,1=good; False = inverted in source)
_GOOD_HIGH = {
    "v2x_polyarchy": True,
    "v2x_libdem": True,
    "v2x_execorrup": True,   # 1 = clean executive
    "v2juncind": True,        # 1 = independent judiciary
    "v2lgello": True,
}


def main() -> None:
    try:
        import vdemdata as vdem
        import pandas as pd
    except ImportError:
        logger.error("Missing packages. Install with: pip install vdemdata pandas")
        sys.exit(1)

    logger.info("Loading V-Dem dataset (may take 30-60s on first run)...")
    df = vdem.load_country_year()

    # Keep only columns we need
    available = [c for c in _COLS if c in df.columns]
    missing = [c for c in _COLS if c not in df.columns]
    if missing:
        logger.warning("Missing columns (check vdemdata version): %s", missing)

    df = df[available].copy()

    # Keep most recent year per country
    latest_year = df["year"].max()
    logger.info("Latest year in dataset: %d", latest_year)
    df = df[df["year"] == latest_year]

    # Normalize all indicator columns to [0, 1]
    ind_cols = [c for c in available if c not in ("country_name", "year")]
    for col in ind_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[col] = 0.5

    # Build snapshot dict
    countries: dict[str, dict] = {}
    for _, row in df.iterrows():
        country = row.get("country_name")
        if not country:
            continue
        entry: dict = {}
        for col in ind_cols:
            v = row.get(col)
            if v is not None and not (v != v):  # not NaN
                entry[col] = round(float(v), 4)
        countries[str(country)] = entry

    snapshot = {
        "year": int(latest_year),
        "source": "V-Dem Dataset v14 (Coppedge et al. 2024)",
        "columns": ind_cols,
        "countries": countries,
    }

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))
    logger.info("Saved %d countries to %s", len(countries), _OUT)


if __name__ == "__main__":
    main()
