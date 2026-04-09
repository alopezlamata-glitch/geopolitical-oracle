from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp

from .base import EvidenceBlock, TTLCache, graceful_collector

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_TIMEOUT = aiohttp.ClientTimeout(total=25)
_CACHE = TTLCache(ttl_minutes=60)   # longer TTL — daily data changes slowly

_STOP_WORDS = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were", "in", "on",
    "at", "to", "for", "of", "and", "or", "by", "with", "this", "that",
    "from", "before", "after", "during", "about", "have", "has", "had",
    "do", "does", "did", "not", "no", "its", "it", "there", "their",
    "happen", "occur", "take", "place", "until",
}


def _cache_key(query: str) -> str:
    tokens = re.sub(r"[^\w\s]", "", query.lower()).split()
    return "gdelt_ts:" + "_".join(tokens[:3])


def _keywords(query: str) -> str:
    tokens = re.sub(r"[^\w\s]", "", query.lower()).split()
    filtered = [t for t in tokens if t not in _STOP_WORDS and len(t) >= 3]
    return " ".join(filtered[:4])


def _gdelt_datetime(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _parse_timeline_series(data) -> list[float]:
    """
    Extract data values from a GDELT timeline API response.
    Actual structure: {"timeline": [{"series": "Volume Intensity", "data": [...]}]}
    """
    try:
        if isinstance(data, str):
            import json as _j
            data = _j.loads(data)
        if not isinstance(data, dict):
            return []
        timeline = data.get("timeline", [])
        if not timeline:
            return []
        # Each entry in timeline is {"series": "name", "data": [{"date":..,"value":..}]}
        first = timeline[0]
        data_points = first.get("data", [])
        return [float(pt["value"]) for pt in data_points if "value" in pt]
    except (KeyError, IndexError, TypeError, ValueError, Exception):
        return []


@graceful_collector("gdelt_timeseries")
async def collect_gdelt_timeseries(
    session: aiohttp.ClientSession, query: str
) -> Optional[EvidenceBlock]:
    """
    Fetch 30-day GDELT time series (volume + tone) for temporal analysis.
    Returns an EvidenceBlock with raw series in metadata for pipeline/temporal.py.
    """
    key = _cache_key(query)
    cached = _CACHE.get(key)
    if cached is not None:
        logger.debug("gdelt_timeseries: cache hit")
        return cached

    keywords = _keywords(query)
    if not keywords:
        return None

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=30)
    start_str = _gdelt_datetime(start)
    end_str = _gdelt_datetime(now)

    base_params = {
        "query": keywords,
        "startdatetime": start_str,
        "enddatetime": end_str,
        "format": "json",
        "timelinesmooth": "3",   # 3-day smoothing to reduce noise
    }

    # Fetch volume and tone series concurrently via separate requests
    vol_params = {**base_params, "mode": "timelinevol"}
    tone_params = {**base_params, "mode": "timelinetone"}

    vol_series: list[float] = []
    tone_series: list[float] = []

    for i, (mode, params, store) in enumerate([
        ("timelinevol", vol_params, None),
        ("timelinetone", tone_params, None),
    ]):
        try:
            if i > 0:
                import asyncio as _asyncio
                await _asyncio.sleep(6)  # GDELT rate limit: 1 req/5s
            async with session.get(_BASE_URL, params=params, timeout=_TIMEOUT) as resp:
                if resp.status == 429:
                    logger.warning("gdelt_timeseries: rate-limited (429) for %s", mode)
                    continue
                resp.raise_for_status()
                raw = await resp.text(encoding="utf-8", errors="replace")
                if not raw or raw.lstrip().startswith("<") or raw.lstrip().startswith("Please"):
                    logger.debug("gdelt_timeseries: non-JSON response for %s: %s", mode, raw[:80])
                    continue
                data = json.loads(raw)
                # Timeline API returns list at top level, not dict
                if isinstance(data, str):
                    data = json.loads(data)  # double-encoded edge case
                parsed = _parse_timeline_series(data)
                if mode == "timelinevol":
                    vol_series = parsed
                else:
                    tone_series = parsed
        except Exception as e:
            logger.warning("gdelt_timeseries: %s failed: %s", mode, e)

    if not vol_series:
        return EvidenceBlock(
            source="gdelt_timeseries",
            content="GDELT 30-day time series unavailable.",
            quality="insufficient",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        )

    # Pad tone series if shorter than volume series
    if len(tone_series) < len(vol_series):
        tone_series = tone_series + [0.0] * (len(vol_series) - len(tone_series))
    elif len(tone_series) > len(vol_series):
        tone_series = tone_series[:len(vol_series)]

    n = len(vol_series)
    avg_vol = sum(vol_series) / n
    avg_tone = sum(tone_series) / n

    content = (
        f"GDELT 30-day series for '{keywords}': "
        f"{n} data points | avg volume={avg_vol:.1f} | avg tone={avg_tone:.2f}"
    )

    block = EvidenceBlock(
        source="gdelt_timeseries",
        content=content,
        quality="medium" if n >= 10 else "low",
        timestamp=datetime.now(timezone.utc),
        metadata={
            "volume_series": vol_series,
            "tone_series": tone_series,
            "keywords": keywords,
            "n_points": n,
        },
    )
    _CACHE.set(key, block)
    return block
