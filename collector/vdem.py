"""
V-Dem (Varieties of Democracy) loader for political indicators.

V-Dem provides 500+ democracy indicators annually for 200+ countries.
Key indicators used here:
  v2x_polyarchy       — Electoral democracy index (0-1)
  v2x_libdem          — Liberal democracy index (0-1)
  v2x_execorrup       — Executive corruption (0=high corruption, 1=clean)
  v2juncind           — Judicial independence (0=low, 1=high)
  v2lgello            — Effective legislature (0=rubber stamp, 1=independent)

Setup (one-time):
  pip install vdemdata
  python scripts/fetch_vdem.py  → builds data/vdem_snapshot.json

Fallback: if snapshot not available, returns zeros (features absent).
The World Bank WGI already provides some governance proxies via collect_worldbank().
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SNAPSHOT_PATH = Path(__file__).parent.parent / "data" / "vdem_snapshot.json"

# V-Dem columns → feature names (all mapped to [0, 1])
_VDEM_TO_FEATURE: dict[str, str] = {
    "v2x_polyarchy":  "pol_electoral_proximity",  # electoral democracy → proxy for electoral pressure
    "v2x_libdem":     "pol_coalition_stability",   # liberal democracy → coalition resilience
    "v2x_execorrup":  "pol_approval_pressure",     # executive corruption (inverted: low=corrupt=pressure)
    "v2juncind":      "pol_judicial_pressure",     # judicial independence (inverted: low=less independent=more pressure)
    "v2lgello":       "pol_resignation_signals",   # legislature effectiveness (inverted: weak=more regime risk)
}


@lru_cache(maxsize=1)
def _load_snapshot() -> Optional[dict]:
    """Load V-Dem snapshot from disk. Cached after first load."""
    if not _SNAPSHOT_PATH.exists():
        logger.debug("vdem: snapshot not found at %s — run scripts/fetch_vdem.py", _SNAPSHOT_PATH)
        return None
    try:
        data = json.loads(_SNAPSHOT_PATH.read_text())
        logger.info("vdem: loaded snapshot (%d countries, year=%s)", len(data.get("countries", {})), data.get("year", "?"))
        return data
    except Exception as e:
        logger.warning("vdem: failed to load snapshot: %s", e)
        return None


def get_vdem_features(country: str) -> dict[str, float]:
    """
    Return V-Dem political indicators for a country.

    Returns dict with pol_* features in [0, 1].
    Returns empty dict if snapshot not available.

    Note: Some mappings are inverted (low independence → higher pressure signal).
    """
    snapshot = _load_snapshot()
    if snapshot is None:
        return {}

    countries = snapshot.get("countries", {})
    # Try exact, title-case, lowercase
    entry = (
        countries.get(country)
        or countries.get(country.title())
        or countries.get(country.lower())
    )
    if entry is None:
        # Partial match
        cl = country.lower()
        for key, val in countries.items():
            if key.lower() in cl or cl in key.lower():
                entry = val
                break

    if entry is None:
        logger.debug("vdem: no data for country '%s'", country)
        return {}

    features: dict[str, float] = {}
    for vdem_col, feat_name in _VDEM_TO_FEATURE.items():
        val = entry.get(vdem_col)
        if val is None:
            continue
        v = float(val)
        # Invert "pressure" indicators: high independence = low pressure
        if feat_name in ("pol_judicial_pressure", "pol_resignation_signals"):
            v = 1.0 - v
        # Invert approval pressure: high corruption (low execorrup) → high pressure
        if feat_name == "pol_approval_pressure":
            v = 1.0 - v
        features[feat_name] = max(0.0, min(1.0, v))

    return features
