"""
World Bank API collector — economic and governance indicators.

Fetches two groups per country (no API key required):
  Economic: GDP growth, inflation, debt/GDP, current account
  Governance (WGI): political stability, govt effectiveness, rule of law

Results are mapped to eco_* and pol_* feature ranges expected by
base_rate_predictor.py and features/builder.py.

Cache: 24h (indicators change at most monthly/annually).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

_BASE = "https://api.worldbank.org/v2"
_TIMEOUT = aiohttp.ClientTimeout(total=12)
_CACHE_DIR = Path(__file__).parent.parent / "data" / "raw" / "cache"
_CACHE_TTL_H = 24

# World Bank indicators to fetch
_ECONOMIC_INDICATORS: dict[str, str] = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",       # GDP growth % (annual)
    "FP.CPI.TOTL.ZG":    "inflation",          # CPI inflation % (annual)
    "GC.DOD.TOTL.GD.ZS": "debt_pct_gdp",      # Central govt debt % GDP
    "BN.CAB.XOKA.GD.ZS": "current_account",   # Current account % GDP
}
_GOVERNANCE_INDICATORS: dict[str, str] = {
    "PV.EST": "wgi_pol_stability",             # Political stability (-2.5..+2.5)
    "GE.EST": "wgi_gov_effectiveness",          # Govt effectiveness (-2.5..+2.5)
    "RL.EST": "wgi_rule_of_law",               # Rule of law (-2.5..+2.5)
}

# Country name → ISO2 code
_ISO2: dict[str, str] = {
    "afghanistan": "AF", "albania": "AL", "algeria": "DZ", "angola": "AO",
    "argentina": "AR", "armenia": "AM", "australia": "AU", "austria": "AT",
    "azerbaijan": "AZ", "bangladesh": "BD", "belgium": "BE", "bolivia": "BO",
    "brazil": "BR", "burkina faso": "BF", "burma": "MM", "cambodia": "KH",
    "cameroon": "CM", "canada": "CA", "central african republic": "CF",
    "chad": "TD", "chile": "CL", "china": "CN", "colombia": "CO", "cuba": "CU",
    "czech republic": "CZ", "czechia": "CZ", "democratic republic of the congo": "CD",
    "denmark": "DK", "drc": "CD", "ecuador": "EC", "egypt": "EG",
    "ethiopia": "ET", "finland": "FI", "france": "FR", "georgia": "GE",
    "germany": "DE", "ghana": "GH", "greece": "GR", "guatemala": "GT",
    "haiti": "HT", "hungary": "HU", "india": "IN", "indonesia": "ID",
    "iran": "IR", "iraq": "IQ", "ireland": "IE", "israel": "IL", "italy": "IT",
    "japan": "JP", "jordan": "JO", "kazakhstan": "KZ", "kenya": "KE",
    "lebanon": "LB", "libya": "LY", "mali": "ML", "mexico": "MX",
    "moldova": "MD", "morocco": "MA", "mozambique": "MZ", "myanmar": "MM",
    "nepal": "NP", "netherlands": "NL", "new zealand": "NZ", "niger": "NE",
    "nigeria": "NG", "north korea": "KP", "norway": "NO", "pakistan": "PK",
    "palestine": "PS", "peru": "PE", "philippines": "PH", "poland": "PL",
    "portugal": "PT", "romania": "RO", "russia": "RU", "saudi arabia": "SA",
    "senegal": "SN", "somalia": "SO", "south africa": "ZA", "south korea": "KR",
    "south sudan": "SS", "spain": "ES", "sri lanka": "LK", "sudan": "SD",
    "sweden": "SE", "switzerland": "CH", "syria": "SY", "taiwan": "TW",
    "tanzania": "TZ", "thailand": "TH", "turkey": "TR", "turkiye": "TR",
    "uganda": "UG", "ukraine": "UA", "united arab emirates": "AE",
    "united kingdom": "GB", "uk": "GB", "united states": "US", "usa": "US",
    "us": "US", "uzbekistan": "UZ", "venezuela": "VE", "vietnam": "VN",
    "yemen": "YE", "zambia": "ZM", "zimbabwe": "ZW",
}


def _country_to_iso2(country: str) -> Optional[str]:
    return _ISO2.get(country.lower().strip())


def _cache_path(iso2: str) -> Path:
    return _CACHE_DIR / f"worldbank_{iso2.lower()}.json"


def _load_cache(iso2: str) -> Optional[dict]:
    path = _cache_path(iso2)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        ts = datetime.fromisoformat(data["timestamp"])
        if (datetime.now(timezone.utc) - ts).total_seconds() < _CACHE_TTL_H * 3600:
            return data["indicators"]
    except Exception:
        pass
    return None


def _save_cache(iso2: str, indicators: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(iso2).write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "indicators": indicators,
    }))


async def _fetch_indicator(
    session: aiohttp.ClientSession,
    iso2: str,
    indicator_id: str,
) -> Optional[float]:
    """Fetch most recent non-null value for a single indicator."""
    url = f"{_BASE}/country/{iso2}/indicator/{indicator_id}"
    params = {"format": "json", "mrv": "5", "gapfill": "Y"}
    try:
        async with session.get(url, params=params, timeout=_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            body = await resp.json(content_type=None)
            if not isinstance(body, list) or len(body) < 2:
                return None
            entries = body[1] or []
            for entry in entries:
                v = entry.get("value")
                if v is not None:
                    return float(v)
    except Exception as e:
        logger.debug("worldbank: %s/%s error: %s", iso2, indicator_id, e)
    return None


def _normalize(value: float, lo: float, hi: float) -> float:
    """Linear normalization to [0, 1], clamped."""
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _raw_to_features(raw: dict) -> dict[str, float]:
    """Map raw World Bank values to eco_* and pol_* feature ranges [0, 1]."""
    feat: dict[str, float] = {}

    # ── Economic features ─────────────────────────────────────────────────────
    gdp = raw.get("gdp_growth")
    if gdp is not None:
        # GDP growth: -5% → 0.0, 0% → 0.38, +5% → 0.77; positive = momentum
        feat["eco_gdp_momentum"] = _normalize(gdp, -5.0, 8.0)

    inflation = raw.get("inflation")
    if inflation is not None:
        # High inflation → economic stress / volatility
        feat["eco_market_volatility"] = _normalize(abs(inflation), 0.0, 25.0)

    debt = raw.get("debt_pct_gdp")
    if debt is not None:
        # Debt/GDP: 30% → 0.2 (low stress), 100% → 0.67, 150%+ → 1.0
        feat["eco_debt_stress"] = _normalize(debt, 0.0, 150.0)

    ca = raw.get("current_account")
    if ca is not None:
        # Large current account deficit → external vulnerability
        feat["eco_policy_uncertainty"] = _normalize(-ca, -10.0, 10.0)

    # ── Governance / political features ──────────────────────────────────────
    ps = raw.get("wgi_pol_stability")
    if ps is not None:
        # WGI: -2.5 (very unstable) to +2.5 (very stable) → 0..1
        feat["pol_coalition_stability"] = _normalize(ps, -2.5, 2.5)

    ge = raw.get("wgi_gov_effectiveness")
    rl = raw.get("wgi_rule_of_law")
    if ge is not None and rl is not None:
        avg = (ge + rl) / 2.0
        # High govt effectiveness + rule of law → lower political risk
        feat["pol_judicial_pressure"] = _normalize(-avg, -2.5, 2.5)  # inverted: bad governance → pressure

    return feat


async def collect_worldbank(
    session: aiohttp.ClientSession,
    country: str,
) -> dict[str, float]:
    """
    Fetch economic and governance indicators for a country.

    Returns dict with eco_* and pol_* features in [0, 1] range.
    Returns empty dict if country unknown or API unavailable.
    """
    iso2 = _country_to_iso2(country)
    if not iso2:
        logger.debug("worldbank: unknown country '%s'", country)
        return {}

    # Try cache first
    cached = _load_cache(iso2)
    if cached is not None:
        logger.debug("worldbank: cache hit for %s (%s)", country, iso2)
        features = _raw_to_features(cached)
        features["_wb_cached"] = 1.0
        return features

    # Fetch all indicators concurrently
    all_indicators = {**_ECONOMIC_INDICATORS, **_GOVERNANCE_INDICATORS}
    import asyncio
    tasks = {
        alias: _fetch_indicator(session, iso2, ind_id)
        for ind_id, alias in all_indicators.items()
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    raw: dict[str, Optional[float]] = {}
    for alias, result in zip(tasks.keys(), results):
        raw[alias] = result if not isinstance(result, Exception) else None

    n_fetched = sum(1 for v in raw.values() if v is not None)
    logger.info(
        "worldbank[%s/%s]: fetched %d/%d indicators",
        country, iso2, n_fetched, len(all_indicators),
    )

    if n_fetched == 0:
        return {}

    _save_cache(iso2, raw)
    features = _raw_to_features(raw)

    # Store raw values for display/audit
    for k, v in raw.items():
        if v is not None:
            features[f"_wb_{k}"] = round(v, 3)

    return features
