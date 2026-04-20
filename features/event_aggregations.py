"""
Shared event-aggregation primitives.

Used by features/builder.py (CanonicalEvent inputs) and
scripts/update_world_state.py (raw dict/object inputs).

Each primitive is formula-neutral: callers pass type matchers and attribute
names so the shared code handles iteration/collection while each file keeps
its own classification and normalisation logic.

Primitives
----------
partition_windows   — bucket events into N-day windows
count_by_type       — count events per category using a caller-supplied function
collect_unique_sources — deduplicate source domains across events
mean_attr           — arithmetic mean of a numeric attribute
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Optional


def partition_windows(
    events: list,
    now: datetime,
    windows_days: tuple[int, ...] = (7, 30),
    get_ts: Callable = lambda e: getattr(e, "occurred_at", None),
) -> dict[int, list]:
    """
    Partition events into N-day windows relative to now.

    Returns {days: [events_within_that_window]}.

    Each event may appear in multiple windows (e.g., a 5d-old event is in
    both the 7d and 30d windows). Handles naive datetimes by assuming UTC.
    Events with no timestamp are silently skipped.

    Parameters
    ----------
    events      : source list (CanonicalEvent or raw dict/object)
    now         : reference datetime (should be UTC-aware)
    windows_days: tuple of window sizes to compute (default: 7d and 30d)
    get_ts      : callable that extracts a datetime from one event;
                  defaults to getattr(e, "occurred_at", None)
    """
    cutoffs = {d: now - timedelta(days=d) for d in windows_days}
    result: dict[int, list] = {d: [] for d in windows_days}

    for ev in events:
        ts = get_ts(ev)
        if ts is None:
            continue
        # Coerce naive timestamps to UTC (handles raw events from update_world_state.py)
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        for d, cutoff in cutoffs.items():
            if ts >= cutoff:
                result[d].append(ev)

    return result


def count_by_type(
    events: list,
    type_fn: Callable,
) -> dict[str, int]:
    """
    Count events per category using a caller-supplied classification function.

    Parameters
    ----------
    events  : list of events
    type_fn : callable(event) → str category or None/empty-string to skip

    Usage
    -----
    # builder.py (exact-match)
    count_by_type(events_7d, lambda e: e.event_type if e.event_type == "military_action" else None)

    # update_world_state.py (substring-match via _event_category)
    count_by_type(events_7d, lambda e: _event_category(getattr(e, "event_type", "")))
    """
    counts: dict[str, int] = {}
    for ev in events:
        cat = type_fn(ev)
        if cat:
            counts[cat] = counts.get(cat, 0) + 1
    return counts


def collect_unique_sources(events: list) -> set[str]:
    """
    Collect unique source domains across all events.

    Prefers .source_domains (list[str]) when present and non-empty.
    Falls back to .source (str) when source_domains is absent or empty.
    Events with neither attribute are silently skipped.
    """
    sources: set[str] = set()
    for ev in events:
        domains = getattr(ev, "source_domains", None)
        if domains:
            sources.update(d for d in domains if d)
        else:
            src = getattr(ev, "source", None)
            if src:
                sources.add(src)
    return sources


def mean_attr(
    events: list,
    attr: str,
    default: float = 0.0,
    cap: Optional[float] = None,
    fallback_attr: Optional[str] = None,
    fallback_scale: float = 1.0,
) -> float:
    """
    Arithmetic mean of a numeric attribute across events.

    Parameters
    ----------
    events         : list of events
    attr           : primary attribute name (e.g. "independent_sources")
    default        : value to return when no events yield a valid value
    cap            : optional upper bound applied after averaging
    fallback_attr  : optional secondary attribute tried when attr is None
    fallback_scale : multiplier applied to the fallback value

    Usage
    -----
    # builder.py — no cap
    mean_attr(events_7d, "independent_sources")

    # update_world_state.py — cap at 5.0, fallback to 1
    mean_attr(events_7d, "independent_sources", default=1.0, cap=5.0)
    """
    vals: list[float] = []
    for ev in events:
        v = getattr(ev, attr, None)
        if v is None and fallback_attr:
            raw = getattr(ev, fallback_attr, None)
            if raw is not None:
                v = float(raw) * fallback_scale
        if v is not None:
            vals.append(float(v))
    if not vals:
        return default
    result = sum(vals) / len(vals)
    if cap is not None:
        result = min(cap, result)
    return result
