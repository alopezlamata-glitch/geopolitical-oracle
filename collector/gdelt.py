from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp

from .base import RawEvent, new_event_id, graceful_collector, load_cached, save_cached, save_raw_events

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_TIMEOUT = aiohttp.ClientTimeout(total=20)

_STOP_WORDS = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were", "in", "on",
    "at", "to", "for", "of", "and", "or", "by", "with", "this", "that",
    "from", "before", "after", "during", "about", "have", "has", "had",
    "do", "does", "did", "not", "no", "its", "it", "there", "their",
    "happen", "occur", "take", "place", "until", "between",
}


def _keywords(query: str) -> str:
    tokens = re.sub(r"[^\w\s]", " ", query.lower()).split()
    filtered = [t for t in tokens if t not in _STOP_WORDS and len(t) >= 3]
    return " ".join(filtered[:5])


def _gdelt_datetime(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _parse_seendate(s: str) -> datetime:
    """Parse GDELT seendate: YYYYMMDDTHHMMSSZ"""
    try:
        s = s.replace("T", "").replace("Z", "")
        return datetime.strptime(s[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _parse_article(art: dict) -> RawEvent:
    actors = []
    allnames = art.get("allnames", "") or ""
    for part in allnames.split(";"):
        part = part.strip()
        if "," in part:
            name = part.split(",")[0].strip()
            if name and len(name) > 2:
                actors.append(name)
    actors = actors[:5]

    themes = []
    themes_raw = art.get("themes", "") or ""
    for t in themes_raw.split(";"):
        t = t.strip()
        if t:
            themes.append(t)
    themes = themes[:10]

    tone_raw = art.get("tone", "") or ""
    try:
        tone = float(str(tone_raw).split(",")[0])
    except (ValueError, IndexError):
        tone = 0.0

    return RawEvent(
        event_id=new_event_id(),
        source="gdelt",
        published_at=_parse_seendate(art.get("seendate", "")),
        title=(art.get("title", "") or "").encode("ascii", "ignore").decode(),
        url=art.get("url", "") or "",
        tone=tone,
        country=art.get("sourcecountry", "") or "",
        event_type="",           # filled during normalization
        sub_event_type="",
        actors=actors,
        fatalities=0,
        themes=themes,
        notes="",
        location={},
        raw_metadata={
            "domain": art.get("domain", ""),
            "language": art.get("language", ""),
        },
    )


@graceful_collector("gdelt")
async def collect_gdelt(session: aiohttp.ClientSession, query: str) -> list[RawEvent]:
    """Fetch last 7 days of GDELT articles matching query keywords."""
    keywords = _keywords(query)
    if not keywords:
        return []

    cache_key = "gdelt_" + re.sub(r"\s+", "_", keywords)[:40]
    cached = load_cached(cache_key)
    if cached is not None:
        logger.debug("gdelt: cache hit (%d events)", len(cached))
        return cached

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=7)

    params = {
        "query": keywords,
        "mode": "artlist",
        "maxrecords": "50",
        "startdatetime": _gdelt_datetime(start),
        "enddatetime": _gdelt_datetime(now),
        "format": "json",
    }

    async with session.get(_BASE_URL, params=params, timeout=_TIMEOUT) as resp:
        if resp.status == 429:
            logger.warning("gdelt: rate-limited (429)")
            return []
        resp.raise_for_status()
        raw = await resp.text(encoding="utf-8", errors="replace")

    if not raw or raw.lstrip().startswith("<") or raw.lstrip().startswith("Please"):
        logger.debug("gdelt: non-JSON response: %s", raw[:80])
        return []

    import json
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("gdelt: JSON parse error: %s", e)
        return []

    articles = data.get("articles", [])
    if not isinstance(articles, list):
        return []

    events = [_parse_article(art) for art in articles if isinstance(art, dict)]
    logger.info("gdelt: collected %d events for query '%s'", len(events), keywords)

    if events:
        save_raw_events("gdelt", events)
        save_cached(cache_key, events)

    return events
