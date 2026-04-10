from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp

from .base import RawEvent, new_event_id, graceful_collector, load_cached, save_cached, save_raw_events

logger = logging.getLogger(__name__)

_BASE_URLS = (
    # Current documented endpoint
    "https://acleddata.com/api/acled/read",
    # Legacy endpoint kept as fallback for compatibility
    "https://api.acleddata.com/acled/read",
)
_TIMEOUT = aiohttp.ClientTimeout(total=15)

# Simple country wordlist for NER from question text
_COUNTRY_ALIASES: dict[str, str] = {
    "iran": "Iran", "iranian": "Iran",
    "israel": "Israel", "israeli": "Israel",
    "russia": "Russia", "russian": "Russia",
    "ukraine": "Ukraine", "ukrainian": "Ukraine",
    "china": "China", "chinese": "China",
    "taiwan": "Taiwan", "taiwanese": "Taiwan",
    "usa": "United States", "us ": "United States", "american": "United States",
    "north korea": "North Korea", "dprk": "North Korea",
    "south korea": "South Korea",
    "pakistan": "Pakistan", "india": "India",
    "syria": "Syria", "syrian": "Syria",
    "yemen": "Yemen", "yemeni": "Yemen",
    "sudan": "Sudan", "ethiopia": "Ethiopia",
    "myanmar": "Myanmar", "burma": "Burma",
    "venezuela": "Venezuela", "haiti": "Haiti",
    "afghanistan": "Afghanistan", "iraq": "Iraq",
    "libya": "Libya", "somalia": "Somalia",
    "mali": "Mali", "niger": "Niger",
    "gaza": "Palestine", "west bank": "Palestine", "palestine": "Palestine",
    "lebanon": "Lebanon", "hezbollah": "Lebanon",
}


def _extract_country(query: str) -> Optional[str]:
    q_lower = query.lower()
    # multi-word first
    for alias, country in sorted(_COUNTRY_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in q_lower:
            return country
    return None


def _parse_event(row: dict) -> RawEvent:
    actors = []
    for field in ("actor1", "actor2"):
        v = row.get(field, "") or ""
        if v and v != "Unknown":
            actors.append(v)

    try:
        fatalities = int(row.get("fatalities", 0) or 0)
    except (ValueError, TypeError):
        fatalities = 0

    date_str = row.get("event_date", "") or ""
    try:
        pub = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        pub = datetime.now(timezone.utc)

    lat = row.get("latitude", None)
    lon = row.get("longitude", None)

    return RawEvent(
        event_id=new_event_id(),
        source="acled",
        published_at=pub,
        title=row.get("notes", "")[:200] or row.get("event_type", ""),
        url="",
        tone=0.0,
        country=row.get("country", ""),
        event_type=row.get("event_type", ""),
        sub_event_type=row.get("sub_event_type", ""),
        actors=actors,
        fatalities=fatalities,
        themes=[],
        notes=row.get("notes", "")[:400],
        location={
            "country": row.get("country", ""),
            "latitude": float(lat) if lat else None,
            "longitude": float(lon) if lon else None,
            "location": row.get("location", ""),
        },
        raw_metadata={
            "source_scale": row.get("source_scale", ""),
            "disorder_type": row.get("disorder_type", ""),
        },
    )


@graceful_collector("acled")
async def collect_acled(session: aiohttp.ClientSession, query: str, country: Optional[str] = None) -> list[RawEvent]:
    access_token = os.environ.get("ACLED_ACCESS_TOKEN", "")
    api_key = os.environ.get("ACLED_API_KEY", "")
    email = os.environ.get("ACLED_EMAIL", "")
    if not access_token and (not api_key or not email):
        logger.debug("acled: no credentials configured (need ACLED_ACCESS_TOKEN or ACLED_API_KEY+ACLED_EMAIL), skipping")
        return []

    target_country = country or _extract_country(query)
    if not target_country:
        logger.debug("acled: could not identify country from query")
        return []

    cache_key = "acled_" + re.sub(r"\s+", "_", target_country.lower())[:30]
    cached = load_cached(cache_key)
    if cached is not None:
        logger.debug("acled: cache hit (%d events)", len(cached))
        return cached

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=30)

    params = {
        "country": target_country,
        "limit": "200",
        "event_date": f"{start.strftime('%Y-%m-%d')}|{now.strftime('%Y-%m-%d')}",
        "event_date_where": "BETWEEN",
        "fields": "event_date|event_type|sub_event_type|actor1|actor2|country|location|fatalities|notes|latitude|longitude|source_scale|disorder_type",
    }
    headers: dict[str, str] = {}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    else:
        # Legacy auth mode
        params["key"] = api_key
        params["email"] = email

    data = None
    for base_url in _BASE_URLS:
        try:
            async with session.get(base_url, params=params, headers=headers, timeout=_TIMEOUT) as resp:
                if resp.status in (401, 403):
                    logger.warning("acled: %s (invalid credentials/token?)", resp.status)
                    return []
                resp.raise_for_status()
                data = await resp.json()
                break
        except aiohttp.ClientError as e:
            logger.warning("acled: request failed for %s: %s", base_url, e)
            continue

    if data is None:
        return []

    rows = data.get("data", [])
    if not isinstance(rows, list):
        return []

    events = [_parse_event(row) for row in rows if isinstance(row, dict)]
    logger.info("acled: collected %d events for country '%s'", len(events), target_country)

    if events:
        save_raw_events("acled", events)
        save_cached(cache_key, events)

    return events
