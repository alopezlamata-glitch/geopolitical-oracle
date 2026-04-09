from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from .base import EvidenceBlock, graceful_collector

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.metaculus.com/api2/questions/"
_MIN_FORECASTERS = 30
_TIMEOUT = aiohttp.ClientTimeout(total=8)


@graceful_collector("metaculus")
async def collect_metaculus(session: aiohttp.ClientSession, query: str) -> EvidenceBlock | None:
    params = {
        "search": query,
        "status": "active",
        "limit": 20,
        "type": "forecast",     # binary/continuous forecast questions only
    }
    headers = {"Accept": "application/json", "User-Agent": "geopolitical-oracle/1.0"}

    async with session.get(_BASE_URL, params=params, headers=headers, timeout=_TIMEOUT) as resp:
        resp.raise_for_status()
        data = await resp.json()

    questions = data.get("results", [])
    if not questions:
        return EvidenceBlock(
            source="metaculus",
            content="No Metaculus questions found for this query.",
            quality="insufficient",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        )

    # Score each question: prefer more forecasters + closer title match
    def score(q: dict) -> float:
        fc = q.get("number_of_forecasters") or 0
        has_pred = _extract_prediction(q) is not None
        return fc * (2.0 if has_pred else 0.5)

    best = max(questions, key=score)
    num_forecasters = best.get("number_of_forecasters") or 0

    # Title can live at multiple depths depending on API version
    title = (
        best.get("title")
        or best.get("question", {}).get("title")
        or best.get("page_url", "Unknown")
    )
    close_time = (
        best.get("close_time")
        or best.get("question", {}).get("close_time")
        or best.get("resolve_time")
        or ""
    )

    cp = _extract_prediction(best)

    if num_forecasters < _MIN_FORECASTERS or cp is None:
        reason = (
            f"only {num_forecasters} forecasters (need >{_MIN_FORECASTERS})"
            if num_forecasters < _MIN_FORECASTERS
            else "no community prediction available"
        )
        return EvidenceBlock(
            source="metaculus",
            content=f"Metaculus: '{title}' — {reason}.",
            quality="insufficient",
            timestamp=datetime.now(timezone.utc),
            metadata={"probability": None, "forecasters": num_forecasters, "close_time": close_time},
        )

    content = (
        f"Metaculus: '{title}'\n"
        f"Community prediction: {cp:.1%} | Forecasters: {num_forecasters} | Closes: {close_time}"
    )

    return EvidenceBlock(
        source="metaculus",
        content=content,
        quality="high" if num_forecasters > 100 else "medium",
        timestamp=datetime.now(timezone.utc),
        metadata={"probability": cp, "forecasters": num_forecasters, "close_time": close_time},
    )


def _extract_prediction(question: dict) -> Optional[float]:
    """
    Extract the community median probability from a Metaculus question object.
    The API has changed shape multiple times — this tries every known path.
    """
    # Paths to check in order of reliability
    candidates = [
        # v2 API — binary questions
        question.get("community_prediction"),
        question.get("question", {}).get("community_prediction"),
        # Sometimes nested under "aggregations"
        question.get("aggregations", {}).get("recency_weighted", {}).get("latest", {}).get("centers", [None])[0]
        if question.get("aggregations") else None,
    ]

    for cp in candidates:
        if cp is None:
            continue

        # Direct float (some API versions return probability directly)
        if isinstance(cp, (int, float)):
            v = float(cp)
            if 0.0 <= v <= 1.0:
                return v

        if isinstance(cp, dict):
            # Try common nested key paths
            for keys in [("full", "q2"), ("q2",), ("median",), ("mean",)]:
                val = cp
                for k in keys:
                    if not isinstance(val, dict):
                        val = None
                        break
                    val = val.get(k)
                if val is not None:
                    try:
                        v = float(val)
                        if 0.0 <= v <= 1.0:
                            return v
                    except (TypeError, ValueError):
                        continue

    return None
