from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_outcome_by_prediction_id(
    prediction_id: str,
    outcome: int,
    resolver_source: str = "user",
    resolution_notes: str = "",
) -> bool:
    """
    Primary resolution path: DuckDB lakehouse by prediction_id.

    Updates:
      - question_resolutions
      - questions.status
      - feature_snapshots.outcome/outcome_resolved_at
      - predictions.brier_component / was_correct for linked prediction rows
    """
    from data_layer.db import get_db, init_schema

    init_schema()
    db = get_db()

    row = db.execute(
        """
        SELECT p.prediction_id, p.question_id, p.snapshot_id, p.calibrated_prob, q.deadline
        FROM predictions p
        LEFT JOIN questions q ON q.question_id = p.question_id
        WHERE p.prediction_id = ?
        """,
        [prediction_id],
    ).fetchone()

    if not row:
        return False

    _, question_id, snapshot_id, calibrated_prob, deadline = row
    now = datetime.now(timezone.utc)

    if question_id:
        existing = db.execute(
            "SELECT resolution_id FROM question_resolutions WHERE question_id = ? LIMIT 1",
            [question_id],
        ).fetchone()
        if not existing:
            db.execute(
                """
                INSERT INTO question_resolutions (
                    resolution_id, question_id, outcome, resolved_at,
                    deadline_was, resolver_source, resolution_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(uuid.uuid4()),
                    question_id,
                    outcome,
                    now,
                    deadline,
                    resolver_source,
                    resolution_notes,
                ],
            )

    # DuckDB FK limitation: updating a referenced parent row can fail even when
    # non-key columns change. Temporarily detach prediction->snapshot links.
    pred_snapshot_rows = db.execute(
        "SELECT prediction_id, snapshot_id FROM predictions WHERE question_id = ?",
        [question_id],
    ).fetchall() if question_id else []

    if pred_snapshot_rows:
        db.execute("UPDATE predictions SET snapshot_id = NULL WHERE question_id = ?", [question_id])

    try:
        if snapshot_id:
            db.execute(
                """
                UPDATE feature_snapshots
                SET outcome = ?, outcome_resolved_at = ?
                WHERE snapshot_id = ?
                """,
                [outcome, now, snapshot_id],
            )
        elif question_id:
            db.execute(
                """
                UPDATE feature_snapshots
                SET outcome = ?, outcome_resolved_at = ?
                WHERE question_id = ?
                """,
                [outcome, now, question_id],
            )
    finally:
        for pred_id, snap_id in pred_snapshot_rows:
            db.execute(
                "UPDATE predictions SET snapshot_id = ? WHERE prediction_id = ?",
                [snap_id, pred_id],
            )

    db.execute(
        """
        UPDATE predictions
        SET
            brier_component = POW(calibrated_prob - ?, 2),
            was_correct = (CASE WHEN ? = 1 THEN calibrated_prob >= 0.5 ELSE calibrated_prob < 0.5 END)
        WHERE question_id = ?
        """,
        [outcome, outcome, question_id],
    )

    if prediction_id:
        db.execute(
            """
            UPDATE predictions
            SET
                brier_component = POW(calibrated_prob - ?, 2),
                was_correct = (CASE WHEN ? = 1 THEN calibrated_prob >= 0.5 ELSE calibrated_prob < 0.5 END)
            WHERE prediction_id = ?
            """,
            [outcome, outcome, prediction_id],
        )

    logger.info("resolve_outcome_by_prediction_id: resolved prediction_id=%s outcome=%s", prediction_id, outcome)
    return True


def resolve_outcome_legacy_json(prediction_id: str, outcome: int) -> bool:
    """Deprecated fallback: resolve legacy JSON artifacts in data/predictions."""
    pred_dir = Path("data/predictions")
    if not pred_dir.exists():
        return False

    matches = list(pred_dir.glob(f"*{prediction_id}*.json"))
    if not matches:
        matches = [f for f in pred_dir.glob("*.json") if prediction_id in f.stem]
    if not matches:
        return False

    path = matches[0]
    data = json.loads(path.read_text())
    data["resolved"] = True
    data["outcome"] = outcome
    data["legacy"] = True
    data["deprecated"] = True
    path.write_text(json.dumps(data, indent=2))

    training_dir = Path("data/training")
    training_dir.mkdir(parents=True, exist_ok=True)
    training_record = {
        "question": data.get("question"),
        "timestamp": data.get("timestamp"),
        "features": data.get("features", {}),
        "outcome": outcome,
        "legacy": True,
        "deprecated": True,
    }
    (training_dir / path.name).write_text(json.dumps(training_record, indent=2))
    return True
