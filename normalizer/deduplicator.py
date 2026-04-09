from __future__ import annotations

import logging
from collections import defaultdict
from datetime import timedelta
from typing import Iterator

from .canonical import CanonicalEvent

logger = logging.getLogger(__name__)


def _actor_overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa = set(x.lower() for x in a)
    sb = set(x.lower() for x in b)
    return len(sa & sb) / max(len(sa | sb), 1)


def _similarity(ev_a: CanonicalEvent, ev_b: CanonicalEvent) -> float:
    type_match = 1.0 if ev_a.event_type == ev_b.event_type else 0.0
    actor_sim = _actor_overlap(ev_a.actors, ev_b.actors)
    return 0.5 * type_match + 0.5 * actor_sim


def _merge(primary: CanonicalEvent, duplicate: CanonicalEvent) -> CanonicalEvent:
    merged_doc_ids = list(set(primary.doc_ids + duplicate.doc_ids))
    unique_sources = len({primary.source, duplicate.source})
    # Use the higher-severity event as base
    base = primary if primary.severity >= duplicate.severity else duplicate
    return CanonicalEvent(
        event_id=base.event_id,
        doc_ids=merged_doc_ids,
        source=base.source,
        occurred_at=base.occurred_at,
        event_type=base.event_type,
        sub_event_type=base.sub_event_type,
        actors=list(set(primary.actors + duplicate.actors))[:8],
        country=base.country or primary.country or duplicate.country,
        severity=max(primary.severity, duplicate.severity),
        polarity=(primary.polarity + duplicate.polarity) / 2,
        fatalities=max(primary.fatalities, duplicate.fatalities),
        independent_sources=unique_sources,
        contradiction_score=abs(primary.polarity - duplicate.polarity) / 2.0,
        raw_title=base.raw_title,
    )


def deduplicate(events: list[CanonicalEvent], window_hours: int = 48) -> list[CanonicalEvent]:
    """
    Merge events that represent the same real-world occurrence.
    Two events are merged if they share same country, within 48h,
    and similarity > 0.75.
    """
    if not events:
        return []

    # Sort by occurred_at for stable processing
    events = sorted(events, key=lambda e: e.occurred_at)
    window = timedelta(hours=window_hours)

    merged: list[CanonicalEvent] = []
    used = set()

    for i, ev_a in enumerate(events):
        if i in used:
            continue
        cluster = [ev_a]
        for j, ev_b in enumerate(events[i + 1:], start=i + 1):
            if j in used:
                continue
            if ev_b.occurred_at - ev_a.occurred_at > window:
                break
            if ev_a.country and ev_b.country and ev_a.country != ev_b.country:
                continue
            if _similarity(ev_a, ev_b) > 0.75:
                cluster.append(ev_b)
                used.add(j)

        # Merge the cluster into one event
        result = cluster[0]
        for other in cluster[1:]:
            result = _merge(result, other)
        result.independent_sources = len({e.source for e in cluster})
        merged.append(result)
        used.add(i)

    logger.info("deduplicator: %d raw → %d after dedup", len(events), len(merged))
    return merged
