from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import timedelta
from typing import Iterator

from .canonical import CanonicalEvent

logger = logging.getLogger(__name__)

_STOP = {"the", "a", "an", "in", "on", "to", "of", "and", "or", "is", "are",
         "was", "were", "by", "for", "with", "at", "from", "after", "says", "said"}


def _title_tokens(title: str) -> set[str]:
    tokens = re.sub(r"[^\w\s]", " ", title.lower()).split()
    return {t for t in tokens if len(t) >= 4 and t not in _STOP}


def _title_overlap(a: str, b: str) -> float:
    ta, tb = _title_tokens(a), _title_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)


def _actor_overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa = set(x.lower() for x in a)
    sb = set(x.lower() for x in b)
    return len(sa & sb) / max(len(sa | sb), 1)


def _similarity(ev_a: CanonicalEvent, ev_b: CanonicalEvent) -> float:
    type_match = 1.0 if ev_a.event_type == ev_b.event_type else 0.0
    actor_sim = _actor_overlap(ev_a.actors, ev_b.actors)
    title_sim = _title_overlap(ev_a.raw_title, ev_b.raw_title)
    # If actors are available use them; otherwise rely on title overlap
    if ev_a.actors and ev_b.actors:
        return 0.4 * type_match + 0.4 * actor_sim + 0.2 * title_sim
    else:
        return 0.3 * type_match + 0.7 * title_sim


def _merge(cluster: list[CanonicalEvent]) -> CanonicalEvent:
    """Merge a cluster of duplicate events into one, tracking domain-level independence."""
    primary = max(cluster, key=lambda e: e.severity)

    merged_doc_ids = list({did for e in cluster for did in e.doc_ids})
    merged_actors = list({a for e in cluster for a in e.actors})[:8]

    # Domain-level independence: count unique root domains across the cluster.
    # This is stricter than counting collector labels (gdelt/rss/acled) because
    # GDELT and RSS aggregate from thousands of syndicated outlets that are NOT
    # editorially independent. A BBC article re-published on 10 GDELT rows is
    # still one independent observation.
    all_domains: set[str] = set()
    for e in cluster:
        for d in e.source_domains:
            if d:
                all_domains.add(d)
    # Cap at 5; fall back to collector-label count if no URL domains were present
    n_independent = len(all_domains) if all_domains else len({e.source for e in cluster})
    n_independent = min(n_independent, 5)

    avg_polarity = sum(e.polarity for e in cluster) / len(cluster)
    contradiction = max(abs(e.polarity - avg_polarity) for e in cluster)

    return CanonicalEvent(
        event_id=primary.event_id,
        doc_ids=merged_doc_ids,
        source=primary.source,
        occurred_at=primary.occurred_at,
        event_type=primary.event_type,
        sub_event_type=primary.sub_event_type,
        actors=merged_actors,
        country=next((e.country for e in cluster if e.country), ""),
        severity=max(e.severity for e in cluster),
        polarity=avg_polarity,
        fatalities=max(e.fatalities for e in cluster),
        independent_sources=n_independent,
        contradiction_score=min(1.0, contradiction),
        raw_title=primary.raw_title,
        source_domains=list(all_domains),
    )


def deduplicate(events: list[CanonicalEvent], window_hours: int = 48) -> list[CanonicalEvent]:
    """
    Merge events that represent the same real-world occurrence.
    Two events are merged if they share same country, within 48h,
    and similarity > 0.75.
    """
    if not events:
        return []

    events = sorted(events, key=lambda e: e.occurred_at)
    window = timedelta(hours=window_hours)

    merged: list[CanonicalEvent] = []
    used: set[int] = set()

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

        merged.append(_merge(cluster))
        used.add(i)

    logger.info("deduplicator: %d raw → %d after dedup", len(events), len(merged))
    return merged
