"""
Seed resolved market predictions from Manifold Markets (free, open API).

Purpose:
  1. Blend calibration — writes (p_model_raw, was_correct) to predictions table
     so blend_calibrator.py can compute Brier comparisons immediately.
  2. Evaluation baseline — resolved rows appear in calibration_audit view.

This does NOT write feature_snapshots, so XGBoost training is unaffected.
Use scripts/seed_training_data.py → python main.py train for XGBoost bootstrap.

Usage:
  python scripts/seed_from_markets.py [--limit 500] [--topic politics]
  python scripts/seed_from_markets.py --dry-run

Manifold API is public — no token needed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_markets")

_MANIFOLD_BASE = "https://api.manifold.markets/v0"
_BATCH_SIZE    = 100
_TOPICS = {
    "politics": [
        "election", "president", "prime minister", "minister", "vote", "resign",
        "parliament", "congress", "senate", "chancellor", "coalition", "party",
        "sanctions", "legislation", "referendum", "impeach",
    ],
    "conflict": [
        "war", "military", "troops", "invasion", "attack", "ceasefire", "peace",
        "missile", "offensive", "occupation", "escalat", "nato", "ukraine",
        "russia", "israel", "hamas", "gaza", "taiwan", "china",
    ],
    "geopolitical": [
        "nato", "united nations", "security council", "treaty", "alliance",
        "diplomatic", "sanctions", "regime", "coup", "protest",
    ],
}

# Combined keyword set for fast pre-filter
_ALL_KEYWORDS = {kw for kws in _TOPICS.values() for kw in kws}


def _question_id(text: str, created_ms: int) -> str:
    raw = f"{text}|{created_ms}"
    return "q_mfd_" + hashlib.sha256(raw.encode()).hexdigest()[:20]


def _prediction_id(question_id: str) -> str:
    return "pred_mfd_" + hashlib.sha256(question_id.encode()).hexdigest()[:24]


def _is_geopolitical(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in _ALL_KEYWORDS)


def _infer_topic(title: str) -> str:
    t = title.lower()
    for topic, kws in _TOPICS.items():
        if any(kw in t for kw in kws):
            return topic
    return "other"


def _infer_country(title: str) -> str:
    mapping = {
        "ukraine": "Ukraine", "russia": "Russia", "united states": "United States",
        "u.s.": "United States", "biden": "United States", "trump": "United States",
        "israel": "Israel", "hamas": "Palestine", "gaza": "Palestine",
        "china": "China", "taiwan": "Taiwan", "xi": "China",
        "ukraine": "Ukraine", "zelensky": "Ukraine", "putin": "Russia",
        "germany": "Germany", "france": "France", "uk": "United Kingdom",
        "britain": "United Kingdom", "iran": "Iran", "north korea": "North Korea",
        "india": "India", "pakistan": "Pakistan", "turkey": "Turkey",
        "saudi": "Saudi Arabia", "nato": "NATO",
    }
    t = title.lower()
    for kw, country in mapping.items():
        if kw in t:
            return country
    return ""


async def _fetch_manifold_batch(
    session, limit: int, before: Optional[str] = None
) -> list[dict]:
    import aiohttp
    url = f"{_MANIFOLD_BASE}/markets"
    params: dict = {
        "filter": "resolved",
        "limit": min(limit, _BATCH_SIZE),
        "sort": "resolve-time",
    }
    if before:
        params["before"] = before

    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                logger.warning("manifold API returned %d", resp.status)
                return []
            return await resp.json()
    except Exception as e:
        logger.warning("manifold fetch failed: %s", e)
        return []


async def _fetch_all(limit: int) -> list[dict]:
    import aiohttp
    results = []
    before = None
    async with aiohttp.ClientSession(headers={
        "User-Agent": "geopolitical-oracle/1.0 (research)"
    }) as session:
        while len(results) < limit:
            batch = await _fetch_manifold_batch(session, _BATCH_SIZE, before)
            if not batch:
                break
            results.extend(batch)
            logger.info("fetched %d / %d Manifold markets", len(results), limit)
            before = batch[-1].get("id")
            if len(batch) < _BATCH_SIZE:
                break
    return results[:limit]


def _parse_market(m: dict) -> Optional[dict]:
    mechanism = m.get("mechanism", "")
    if mechanism != "binary":
        return None

    resolution = m.get("resolution", "")
    if resolution not in ("YES", "NO"):
        return None  # ignore CANCEL / MKT

    title = m.get("question", "").strip()
    if not title or not _is_geopolitical(title):
        return None

    outcome = 1 if resolution == "YES" else 0

    # Community probability just before close (may not always be available)
    prob = m.get("probability")
    if prob is None or not (0.01 <= prob <= 0.99):
        return None

    close_ms   = m.get("closeTime") or m.get("resolutionTime") or 0
    created_ms = m.get("createdTime") or 0
    resolve_ms = m.get("resolutionTime") or close_ms

    deadline_dt = datetime.fromtimestamp(close_ms / 1000, tz=timezone.utc) if close_ms else datetime.now(timezone.utc)
    resolved_dt = datetime.fromtimestamp(resolve_ms / 1000, tz=timezone.utc) if resolve_ms else datetime.now(timezone.utc)
    created_dt  = datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc) if created_ms else datetime.now(timezone.utc)

    market_id  = m.get("id", "")
    country    = _infer_country(title)
    topic      = _infer_topic(title)
    volume     = float(m.get("volume", 0) or 0)
    traders    = int(m.get("uniqueBettorCount", 0) or 0)

    return {
        "title":       title,
        "outcome":     outcome,
        "probability": float(prob),
        "deadline":    deadline_dt,
        "resolved_at": resolved_dt,
        "created_at":  created_dt,
        "market_id":   market_id,
        "country":     country,
        "topic":       topic,
        "volume":      volume,
        "traders":     traders,
        "created_ms":  created_ms,
    }


def _write_to_db(records: list[dict], dry_run: bool = False) -> tuple[int, int]:
    from data_layer.db import get_db, init_schema, table_exists

    init_schema()
    db = get_db()

    n_questions  = 0
    n_predictions = 0

    for rec in records:
        q_id   = _question_id(rec["title"], rec["created_ms"])
        pred_id = _prediction_id(q_id)

        if dry_run:
            logger.info(
                "[DRY] %s → outcome=%d  p=%.3f  country=%s",
                rec["title"][:60], rec["outcome"], rec["probability"], rec["country"],
            )
            n_questions += 1
            n_predictions += 1
            continue

        # ── Write question record ──────────────────────────────────────────────
        try:
            db.execute(
                """
                INSERT OR IGNORE INTO questions (
                    question_id, raw_text, subject, predicate, event_family,
                    jurisdiction, deadline, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    q_id,
                    rec["title"][:500],
                    rec["country"] or rec["topic"],
                    rec["topic"],
                    rec["topic"],
                    rec["country"] or None,
                    rec["deadline"].date(),
                    "resolved",
                    rec["created_at"],
                ],
            )
            n_questions += 1
        except Exception as e:
            logger.debug("question insert failed for %s: %s", q_id, e)

        # ── Write question_resolution ──────────────────────────────────────────
        try:
            res_id = "res_mfd_" + hashlib.sha256(q_id.encode()).hexdigest()[:24]
            db.execute(
                """
                INSERT OR IGNORE INTO question_resolutions (
                    resolution_id, question_id, outcome,
                    resolved_at, deadline_was,
                    resolver_source, resolution_notes,
                    resolution_confidence, is_ambiguous, resolved_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    res_id, q_id, rec["outcome"],
                    rec["resolved_at"], rec["deadline"].date(),
                    rec.get("source", "manifold"),
                    json.dumps({"market_id": rec["market_id"], "volume": rec["volume"], "traders": rec["traders"]}),
                    1.0, False, "seed_from_markets",
                ],
            )
        except Exception as e:
            logger.debug("resolution insert failed for %s: %s", q_id, e)

        # ── Write prediction row (for blend calibration) ───────────────────────
        try:
            was_correct = bool(rec["outcome"] == 1) == (rec["probability"] >= 0.5)
            db.execute(
                """
                INSERT OR IGNORE INTO predictions (
                    prediction_id, question_id, snapshot_id,
                    raw_prob, calibrated_prob, answer,
                    model_id, model_version, schema_version,
                    p_model_raw, p_market_raw, market_weight,
                    blend_strategy, blend_strategy_version,
                    n_market_signals, market_gate_passed,
                    brier_component, was_correct,
                    predicted_at, as_of_time
                ) VALUES (
                    ?, ?, NULL,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, NULL, 0.0,
                    ?, ?,
                    0, FALSE,
                    ?, ?,
                    ?, ?
                )
                """,
                [
                    pred_id, q_id,
                    rec["probability"], rec["probability"],
                    "YES" if rec["probability"] >= 0.5 else "NO",
                    "manifold_community_seed", "1.0", "manifold_seed_v1",
                    rec["probability"],  # p_model_raw = manifold community p
                    "model_only", "logodds_v1",
                    round((rec["probability"] - rec["outcome"]) ** 2, 6),
                    was_correct,
                    rec["resolved_at"], rec["deadline"],
                ],
            )
            n_predictions += 1
        except Exception as e:
            logger.debug("prediction insert failed for %s: %s", pred_id, e)

    return n_questions, n_predictions


_METACULUS_BASE = "https://www.metaculus.com/api2"


async def _fetch_metaculus_resolved(
    session, limit: int = 200, token: str = ""
) -> list[dict]:
    """
    Fetch resolved binary questions from Metaculus v2 API.
    For resolved questions, community_prediction.full.q2 is visible even without auth.
    Requires METACULUS_API_TOKEN for full access; partial data available without.
    """
    import aiohttp
    headers = {"User-Agent": "geopolitical-oracle/1.0 (research)", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"

    url = f"{_METACULUS_BASE}/questions/"
    params = {
        "type":        "binary",
        "status":      "resolved",
        "order_by":    "-resolve_time",
        "limit":       min(limit, 100),
        "offset":      0,
    }
    results = []
    try:
        while len(results) < limit:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20), headers=headers) as resp:
                if resp.status == 403:
                    logger.debug("metaculus: 403 (token required or rate limited)")
                    break
                if resp.status != 200:
                    logger.warning("metaculus: HTTP %d", resp.status)
                    break
                data = await resp.json()
                items = data.get("results", [])
                if not items:
                    break
                results.extend(items)
                logger.info("metaculus: fetched %d / %d", len(results), limit)
                if not data.get("next"):
                    break
                params["offset"] = params["offset"] + len(items)
    except Exception as e:
        logger.warning("metaculus fetch failed: %s", e)

    return results[:limit]


def _parse_metaculus(m: dict) -> Optional[dict]:
    """Parse a resolved Metaculus question into our canonical record format."""
    resolution = m.get("resolution")
    if resolution not in (1.0, 0.0, 1, 0):
        return None

    title = (m.get("title") or "").strip()
    if not title or not _is_geopolitical(title):
        return None

    # Community prediction (visible for resolved questions)
    cp = m.get("community_prediction") or {}
    full = cp.get("full") or {}
    prob = full.get("q2")  # median probability

    # Fallback: use resolution itself as probability signal (at close)
    if prob is None or not (0.01 <= float(prob) <= 0.99):
        return None

    outcome = 1 if int(resolution) == 1 else 0
    close_time   = m.get("close_time")   or m.get("scheduled_resolve_time")
    resolve_time = m.get("resolve_time") or close_time
    created_time = m.get("created_time")

    def _parse_dt(s) -> datetime:
        if not s:
            return datetime.now(timezone.utc)
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    deadline_dt  = _parse_dt(close_time)
    resolved_dt  = _parse_dt(resolve_time)
    created_dt   = _parse_dt(created_time)

    return {
        "title":       title,
        "outcome":     outcome,
        "probability": float(prob),
        "deadline":    deadline_dt,
        "resolved_at": resolved_dt,
        "created_at":  created_dt,
        "market_id":   f"metaculus_{m.get('id', '')}",
        "country":     _infer_country(title),
        "topic":       _infer_topic(title),
        "volume":      float(m.get("activity") or m.get("effected_prediction_count") or 0),
        "traders":     int(m.get("number_of_forecasters") or m.get("effected_prediction_count") or 0),
        "created_ms":  int(created_dt.timestamp() * 1000),
        "source":      "metaculus",
    }


def run(
    limit: int = 500,
    dry_run: bool = False,
    topic_filter: Optional[str] = None,
    include_metaculus: bool = True,
) -> None:
    import os
    meta_token = os.getenv("METACULUS_API_TOKEN", "").strip()

    logger.info("fetching up to %d resolved Manifold markets...", limit)
    raw_manifold = asyncio.run(_fetch_all(limit))
    logger.info("downloaded %d Manifold markets, parsing...", len(raw_manifold))

    parsed = []
    for m in raw_manifold:
        rec = _parse_market(m)
        if rec is None:
            continue
        if topic_filter and rec["topic"] != topic_filter:
            continue
        rec["source"] = "manifold"
        parsed.append(rec)

    # Also fetch from Metaculus if token available
    if include_metaculus:
        if not meta_token:
            logger.info("metaculus: no METACULUS_API_TOKEN — attempting public endpoint")

        async def _fetch_meta():
            import aiohttp
            async with aiohttp.ClientSession() as session:
                return await _fetch_metaculus_resolved(session, limit=min(limit, 300), token=meta_token)

        raw_meta = asyncio.run(_fetch_meta())
        logger.info("metaculus: downloaded %d resolved questions", len(raw_meta))
        n_meta_parsed = 0
        for m in raw_meta:
            rec = _parse_metaculus(m)
            if rec is None:
                continue
            if topic_filter and rec["topic"] != topic_filter:
                continue
            parsed.append(rec)
            n_meta_parsed += 1
        logger.info("metaculus: kept %d geopolitical binary questions", n_meta_parsed)

    logger.info(
        "total qualifying markets: %d (manifold=%d metaculus=%d)",
        len(parsed),
        sum(1 for r in parsed if r.get("source") == "manifold"),
        sum(1 for r in parsed if r.get("source") == "metaculus"),
    )

    if not parsed:
        logger.warning("no qualifying markets found")
        return

    n_q, n_p = _write_to_db(parsed, dry_run=dry_run)

    print(f"\n{'DRY RUN — ' if dry_run else ''}seed_from_markets complete")
    print(f"  Qualifying markets : {len(parsed)}")
    print(f"  Questions written  : {n_q}")
    print(f"  Predictions written: {n_p}")

    if not dry_run:
        try:
            from data_layer.db import get_db
            db = get_db()
            n_res = db.execute("SELECT COUNT(*) FROM question_resolutions").fetchone()[0]
            n_pred = db.execute("SELECT COUNT(*) FROM predictions WHERE was_correct IS NOT NULL").fetchone()[0]
            print(f"  DB total resolutions: {n_res}")
            print(f"  DB resolved predictions: {n_pred}")
            if n_pred >= 20:
                print("\n  Ready for blend calibration: python main.py blend-calibrate")
        except Exception as e:
            logger.debug("stats query failed: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed resolved market predictions from Manifold + Metaculus")
    parser.add_argument("--limit",  type=int, default=500, help="Max markets to fetch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--topic", choices=list(_TOPICS), help="Filter to one topic category")
    parser.add_argument("--no-metaculus", action="store_true", help="Skip Metaculus (Manifold only)")
    args = parser.parse_args()
    run(limit=args.limit, dry_run=args.dry_run, topic_filter=args.topic,
        include_metaculus=not args.no_metaculus)


if __name__ == "__main__":
    main()
