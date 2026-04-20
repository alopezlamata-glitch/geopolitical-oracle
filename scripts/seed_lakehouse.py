"""
Migrate ICEWS legacy JSON training data into the DuckDB lakehouse.

After running this script, the `training_ready_snapshots` view will be
populated and future training will use the lakehouse natively instead of
falling back to the JSON files.

Run: python scripts/seed_lakehouse.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_lakehouse")

_TRAINING_DIR = Path(__file__).parent.parent / "data" / "training"
_FEATURE_SCHEMA_VER = "v3"
_BUILDER_VERSION = "icews_seed_v1"


def _make_id(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:32]


def _parse_timestamp(ts: str) -> datetime:
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)


def seed(dry_run: bool = False) -> None:
    from data_layer.db import get_db, init_schema

    init_schema()
    db = get_db()

    json_files = sorted(_TRAINING_DIR.glob("*.json"))
    if not json_files:
        logger.error("No JSON files found in %s", _TRAINING_DIR)
        return

    logger.info("Found %d training JSON files", len(json_files))

    # Check existing snapshots to avoid duplicates
    existing = set()
    try:
        rows = db.execute("SELECT snapshot_id FROM feature_snapshots").fetchall()
        existing = {r[0] for r in rows}
        logger.info("%d snapshots already in lakehouse", len(existing))
    except Exception as e:
        logger.warning("Could not query existing snapshots: %s", e)

    inserted = 0
    skipped = 0
    errors = 0
    now = datetime.now(timezone.utc)

    for f in json_files:
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Skipping malformed file %s: %s", f.name, e)
            errors += 1
            continue

        if "outcome" not in rec or "features" not in rec:
            skipped += 1
            continue

        outcome = int(rec["outcome"])
        features: dict = rec["features"]
        question_text: str = rec.get("question", f.stem)
        timestamp: datetime = _parse_timestamp(rec.get("timestamp", now.isoformat()))
        country: str = rec.get("country", "")
        source: str = rec.get("source", "icews_historical")

        # Stable IDs derived from file content (idempotent)
        question_id = _make_id(f"question:{question_text}:{timestamp.isoformat()}")
        snapshot_id = _make_id(f"snapshot:{f.name}")
        resolution_id = _make_id(f"resolution:{question_id}")

        if snapshot_id in existing:
            skipped += 1
            continue

        if dry_run:
            inserted += 1
            continue

        # ── Insert question ──────────────────────────────────────────────────
        try:
            db.execute(
                """
                INSERT OR IGNORE INTO questions (
                    question_id, raw_text, subject, predicate, event_family,
                    jurisdiction, is_negated, deadline, resolution_rule,
                    status, parse_confidence, matched_model,
                    created_at, as_of_time, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    question_id,
                    question_text,
                    country or "unknown",
                    "military_escalation",
                    "conflict",
                    country or None,
                    False,
                    timestamp.date(),                  # deadline = window date
                    "Military intensity >= 30 in 30 days",
                    "resolved",
                    1.0,
                    "xgb_conflict_v3",
                    timestamp,
                    timestamp,
                    source,
                ],
            )
        except Exception as e:
            logger.debug("question insert skipped (%s): %s", question_id[:8], e)

        # ── Insert feature snapshot ──────────────────────────────────────────
        try:
            db.execute(
                """
                INSERT OR IGNORE INTO feature_snapshots (
                    snapshot_id, question_id, as_of_time,
                    question_predicate,
                    explicit_features, feature_schema_ver,
                    builder_version, built_at,
                    outcome, outcome_resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    snapshot_id,
                    question_id,
                    timestamp,
                    "military_escalation",
                    json.dumps(features),
                    _FEATURE_SCHEMA_VER,
                    _BUILDER_VERSION,
                    now,
                    outcome,
                    timestamp,
                ],
            )
        except Exception as e:
            logger.warning("snapshot insert failed (%s): %s", f.name, e)
            errors += 1
            continue

        # ── Insert question resolution ───────────────────────────────────────
        try:
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
                    resolution_id,
                    question_id,
                    outcome,
                    timestamp,
                    timestamp.date(),
                    "icews_historical",
                    f"ICEWS historical window: {f.name}",
                    1.0,
                    False,
                    "seed_lakehouse",
                ],
            )
        except Exception as e:
            logger.debug("resolution insert skipped (%s): %s", resolution_id[:8], e)

        inserted += 1

    total = inserted + skipped + errors
    logger.info(
        "Done. %d inserted, %d skipped (already existed), %d errors / %d total",
        inserted, skipped, errors, total,
    )

    if not dry_run:
        # Verify view
        try:
            n = db.execute("SELECT COUNT(*) FROM training_ready_snapshots").fetchone()[0]
            logger.info("training_ready_snapshots now has %d rows", n)
        except Exception as e:
            logger.warning("Could not count training_ready_snapshots: %s", e)

    if dry_run:
        logger.info("[DRY RUN] Would insert %d rows — no changes made", inserted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed DuckDB lakehouse from ICEWS JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Count rows without inserting")
    args = parser.parse_args()
    seed(dry_run=args.dry_run)
    if not args.dry_run:
        print("Run 'python main.py train' to retrain using lakehouse as primary source.")


if __name__ == "__main__":
    main()
