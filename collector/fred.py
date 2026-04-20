"""
FRED (Federal Reserve Economic Data) collector.

Fetches US monetary policy signals used for economic domain predictions.
Primary use: populate eco_* features for Fed rate decisions, recession risk.

Free API key: https://fred.stlouisfed.org/docs/api/api_key.html
Set in .env: FRED_API_KEY=your_key_here

Non-fatal: returns empty dict if key missing or API unavailable.
Cache: 6 hours (FRED data updates daily at most).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

_BASE = "https://api.stlouisfed.org/fred/series/observations"
_TIMEOUT = aiohttp.ClientTimeout(total=12)
_CACHE_DIR = Path(__file__).parent.parent / "data" / "raw" / "cache"
_CACHE_TTL_H = 6

# Series to fetch: series_id → (alias, description)
_SERIES: dict[str, tuple[str, str]] = {
    "DFF":          ("fed_funds_rate",    "Effective Federal Funds Rate (%)"),
    "T10Y2Y":       ("yield_spread_10y2y","10Y-2Y Treasury Spread (%)"),
    "T10YIE":       ("breakeven_10y",     "10Y Breakeven Inflation Rate (%)"),
    "BAMLH0A0HYM2": ("hy_spread",         "US High Yield OAS (bps / 100)"),
    "VIXCLS":       ("vix",               "CBOE Volatility Index"),
    "CPIAUCSL":     ("cpi_level",         "CPI All Items (index)"),
}


def _cache_path() -> Path:
    return _CACHE_DIR / "fred_us.json"


def _load_cache() -> Optional[dict]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        ts = datetime.fromisoformat(data["timestamp"])
        if (datetime.now(timezone.utc) - ts).total_seconds() < _CACHE_TTL_H * 3600:
            return data["series"]
    except Exception:
        pass
    return None


def _save_cache(series: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path().write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series,
    }))


async def _fetch_series(
    session: aiohttp.ClientSession,
    series_id: str,
    api_key: str,
    n_obs: int = 13,
) -> Optional[list[float]]:
    """Fetch last n_obs observations for a FRED series. Returns list newest-first."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": str(n_obs),
        "observation_start": (datetime.now(timezone.utc) - timedelta(days=400)).strftime("%Y-%m-%d"),
    }
    try:
        async with session.get(_BASE, params=params, timeout=_TIMEOUT) as resp:
            if resp.status == 400:
                # Bad API key or series not found
                text = await resp.text()
                logger.debug("fred: %s 400: %s", series_id, text[:100])
                return None
            if resp.status != 200:
                return None
            body = await resp.json(content_type=None)
            obs = body.get("observations", [])
            values = []
            for o in obs:
                v = o.get("value", ".")
                if v != "." and v is not None:
                    try:
                        values.append(float(v))
                    except (ValueError, TypeError):
                        pass
            return values if values else None
    except Exception as e:
        logger.debug("fred: %s error: %s", series_id, e)
        return None


def _compute_cpi_yoy(cpi_values: list[float]) -> Optional[float]:
    """Compute CPI year-over-year % change from last 13 monthly observations."""
    if len(cpi_values) < 13:
        return None
    return (cpi_values[0] / cpi_values[12] - 1.0) * 100.0


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _raw_to_features(raw: dict) -> dict[str, float]:
    """Map FRED series observations to eco_* feature ranges [0, 1]."""
    feat: dict[str, float] = {}

    fed_funds = raw.get("fed_funds_rate")
    spread = raw.get("yield_spread_10y2y")
    hy = raw.get("hy_spread")
    vix = raw.get("vix")
    cpi_obs = raw.get("cpi_obs")
    breakeven = raw.get("breakeven_10y")

    # ── eco_rate_change_prob ──────────────────────────────────────────────────
    # Signal: rate cut likely if yield curve inverted AND funds rate elevated
    # Signal: rate hike likely if inflation breakeven high AND funds rate low
    if fed_funds is not None and spread is not None:
        ff = fed_funds[0] if isinstance(fed_funds, list) else fed_funds
        sp = spread[0] if isinstance(spread, list) else spread
        # Inverted curve (sp < 0) + high rates → rate cut signal
        cut_signal = _normalize(-sp, -1.5, 0.5) * _normalize(ff, 0.0, 6.0)
        # Steep curve + low rates → rate hike signal
        hike_signal = _normalize(sp, -0.5, 2.0) * _normalize(4.0 - ff, -2.0, 4.0)
        feat["eco_rate_change_prob"] = min(1.0, cut_signal + hike_signal * 0.5)
        feat["eco_policy_uncertainty"] = _normalize(-sp, -2.0, 1.5)  # inverted = uncertainty

    # ── eco_market_volatility ─────────────────────────────────────────────────
    vix_score = 0.0
    hy_score = 0.0
    if vix is not None:
        v = vix[0] if isinstance(vix, list) else vix
        # VIX: 10=calm, 20=normal, 40=stress, 80=crisis
        vix_score = _normalize(v, 8.0, 60.0)
    if hy is not None:
        h = hy[0] if isinstance(hy, list) else hy
        # HY spread in %: 2=tight, 4=normal, 8=stress, 15=crisis (FRED reports in %)
        hy_score = _normalize(h, 2.0, 10.0)

    if vix_score or hy_score:
        # Weight VIX more (more timely)
        denom = (2 if vix_score else 0) + (1 if hy_score else 0)
        feat["eco_market_volatility"] = (vix_score * 2 + hy_score) / max(1, denom)

    # ── eco_debt_stress ───────────────────────────────────────────────────────
    # Proxy: high HY spreads + inverted curve → financial stress
    if hy is not None:
        h = hy[0] if isinstance(hy, list) else hy
        feat["eco_debt_stress"] = _normalize(h, 2.5, 10.0)

    # ── CPI inflation ─────────────────────────────────────────────────────────
    if isinstance(cpi_obs, list) and cpi_obs:
        cpi_yoy = _compute_cpi_yoy(cpi_obs)
        if cpi_yoy is not None:
            feat["_fred_cpi_yoy"] = round(cpi_yoy, 2)
            # High YoY inflation → market volatility proxy if not already set from VIX
            if "eco_market_volatility" not in feat:
                feat["eco_market_volatility"] = _normalize(abs(cpi_yoy), 0.0, 12.0)

    # Store raw latest values for audit
    for alias in ("fed_funds_rate", "yield_spread_10y2y", "vix", "hy_spread"):
        obs = raw.get(alias)
        if obs:
            v = obs[0] if isinstance(obs, list) else obs
            feat[f"_fred_{alias}"] = round(v, 3)
    if breakeven:
        b = breakeven[0] if isinstance(breakeven, list) else breakeven
        feat["_fred_breakeven_10y"] = round(b, 3)

    return feat


async def collect_fred(session: aiohttp.ClientSession) -> dict[str, float]:
    """
    Fetch US monetary policy signals from FRED.

    Returns dict with eco_* features in [0, 1] + raw _fred_* audit values.
    Returns empty dict if FRED_API_KEY not set or API unavailable.
    """
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        logger.debug("fred: FRED_API_KEY not set, skipping")
        return {}

    cached = _load_cache()
    if cached is not None:
        logger.debug("fred: cache hit")
        return _raw_to_features(cached)

    import asyncio
    raw: dict = {}

    tasks = {
        alias: _fetch_series(session, series_id, api_key)
        for series_id, (alias, _) in _SERIES.items()
        if alias != "cpi_level"
    }
    # CPI needs more observations for YoY
    tasks["cpi_obs"] = _fetch_series(session, "CPIAUCSL", api_key, n_obs=14)

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for alias, result in zip(tasks.keys(), results):
        raw[alias] = result if not isinstance(result, Exception) else None

    n_ok = sum(1 for v in raw.values() if v is not None)
    logger.info("fred: fetched %d/%d series", n_ok, len(tasks))

    if n_ok == 0:
        return {}

    _save_cache(raw)
    return _raw_to_features(raw)
