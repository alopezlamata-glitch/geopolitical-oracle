from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp

from .base import EvidenceBlock, graceful_collector

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.acleddata.com/acled/read/"
_TIMEOUT = aiohttp.ClientTimeout(total=10)
_LOOKBACK_DAYS = 60

# Comprehensive country list for NER extraction from question text
_COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Angola", "Argentina", "Armenia",
    "Australia", "Austria", "Azerbaijan", "Bahrain", "Bangladesh", "Belarus",
    "Belgium", "Bolivia", "Bosnia", "Brazil", "Bulgaria", "Burma", "Cambodia",
    "Cameroon", "Canada", "Chad", "Chile", "China", "Colombia", "Congo",
    "Croatia", "Cuba", "Cyprus", "Czech", "Denmark", "Ecuador", "Egypt",
    "Ethiopia", "Finland", "France", "Georgia", "Germany", "Ghana", "Greece",
    "Guatemala", "Haiti", "Honduras", "Hungary", "India", "Indonesia", "Iran",
    "Iraq", "Ireland", "Israel", "Italy", "Japan", "Jordan", "Kazakhstan",
    "Kenya", "Kosovo", "Kuwait", "Kyrgyzstan", "Lebanon", "Libya", "Malaysia",
    "Mali", "Mexico", "Moldova", "Morocco", "Mozambique", "Myanmar", "Nepal",
    "Netherlands", "Nicaragua", "Nigeria", "North Korea", "Norway", "Pakistan",
    "Palestine", "Panama", "Peru", "Philippines", "Poland", "Portugal",
    "Qatar", "Romania", "Russia", "Rwanda", "Saudi Arabia", "Serbia",
    "Somalia", "South Africa", "South Korea", "South Sudan", "Spain",
    "Sri Lanka", "Sudan", "Sweden", "Switzerland", "Syria", "Taiwan",
    "Tajikistan", "Tanzania", "Thailand", "Tunisia", "Turkey", "Turkmenistan",
    "Ukraine", "United Kingdom", "United States", "Uzbekistan", "Venezuela",
    "Vietnam", "Yemen", "Zimbabwe",
]

# Aliases for common references
_ALIASES = {
    "US": "United States", "USA": "United States", "UK": "United Kingdom",
    "Britain": "United Kingdom", "England": "United Kingdom",
    "Korea": "South Korea", "DPRK": "North Korea",
    "Persia": "Iran", "Taiwan": "Taiwan", "Gaza": "Palestine",
    "West Bank": "Palestine", "Donbas": "Ukraine", "Crimea": "Ukraine",
}


def _extract_countries(text: str) -> list[str]:
    """Extract country names from question text."""
    found = []
    text_lower = text.lower()

    # Check aliases first
    for alias, canonical in _ALIASES.items():
        if alias.lower() in text_lower and canonical not in found:
            found.append(canonical)

    # Check country list
    for country in _COUNTRIES:
        if country.lower() in text_lower and country not in found:
            found.append(country)

    return found[:3]  # limit to top 3 countries


def _is_available() -> bool:
    return bool(os.environ.get("ACLED_API_KEY") and os.environ.get("ACLED_EMAIL"))


@graceful_collector("acled")
async def collect_acled(
    session: aiohttp.ClientSession, query: str
) -> Optional[EvidenceBlock]:
    """
    Fetch recent conflict events from ACLED for countries mentioned in the question.

    Requires env vars: ACLED_API_KEY, ACLED_EMAIL
    (Register free at: https://developer.acleddata.com)

    Skips gracefully if credentials are not set.
    """
    if not _is_available():
        logger.debug("acled: no credentials, skipping")
        return EvidenceBlock(
            source="acled",
            content="ACLED: credentials not configured (set ACLED_API_KEY + ACLED_EMAIL).",
            quality="insufficient",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        )

    countries = _extract_countries(query)
    if not countries:
        return EvidenceBlock(
            source="acled",
            content="ACLED: no countries identified in question.",
            quality="insufficient",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        )

    api_key = os.environ["ACLED_API_KEY"]
    email = os.environ["ACLED_EMAIL"]

    cutoff = datetime.now(timezone.utc) - timedelta(days=_LOOKBACK_DAYS)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    params = {
        "key": api_key,
        "email": email,
        "event_date": f"{cutoff_str}|{today_str}",
        "event_date_where": "BETWEEN",
        "country": "|".join(countries),
        "country_where": "OR",
        "limit": 100,
        "fields": "event_date|event_type|actor1|actor2|fatalities|country|notes",
        "format": "json",
    }

    async with session.get(_BASE_URL, params=params, timeout=_TIMEOUT) as resp:
        resp.raise_for_status()
        data = await resp.json(content_type=None)

    events = data.get("data", [])
    if not events:
        return EvidenceBlock(
            source="acled",
            content=f"ACLED: no conflict events in {', '.join(countries)} in last {_LOOKBACK_DAYS} days.",
            quality="low",
            timestamp=datetime.now(timezone.utc),
            metadata={"countries": countries, "event_count": 0, "fatalities": 0},
        )

    # Aggregate statistics
    total_fatalities = sum(int(e.get("fatalities") or 0) for e in events)
    event_types: dict[str, int] = {}
    for e in events:
        et = e.get("event_type", "Unknown")
        event_types[et] = event_types.get(et, 0) + 1

    top_type = max(event_types, key=lambda k: event_types[k]) if event_types else "Unknown"
    type_breakdown = " | ".join(f"{k}: {v}" for k, v in sorted(event_types.items(), key=lambda x: -x[1])[:4])

    # Sample recent notable events (high fatality or protests)
    notable = sorted(events, key=lambda e: int(e.get("fatalities") or 0), reverse=True)[:3]
    notable_lines = []
    for e in notable:
        actor = e.get("actor1", "?")
        etype = e.get("event_type", "?")
        fat = int(e.get("fatalities") or 0)
        date = e.get("event_date", "?")
        notable_lines.append(f"  • {date}: {etype} involving {actor} ({fat} fatalities)")

    content = (
        f"ACLED conflict data — {', '.join(countries)} (last {_LOOKBACK_DAYS} days):\n"
        f"Total events: {len(events)} | Total fatalities: {total_fatalities}\n"
        f"Event types: {type_breakdown}\n"
        f"Notable events:\n" + "\n".join(notable_lines)
    )

    quality = "high" if total_fatalities > 50 or len(events) > 30 else "medium" if events else "low"

    return EvidenceBlock(
        source="acled",
        content=content,
        quality=quality,
        timestamp=datetime.now(timezone.utc),
        metadata={
            "countries": countries,
            "event_count": len(events),
            "fatalities": total_fatalities,
            "dominant_event_type": top_type,
            "event_type_breakdown": event_types,
        },
    )
