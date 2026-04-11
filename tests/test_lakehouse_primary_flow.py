from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture()
def temp_lakehouse(tmp_path, monkeypatch):
    from data_layer import db as dbmod

    dbmod.close_db()
    test_db = tmp_path / "oracle_test.duckdb"
    monkeypatch.setattr(dbmod, "_DB_PATH", test_db)
    dbmod.init_schema(force=True)
    yield dbmod.get_db()
    dbmod.close_db()


def _seed_prediction_graph(db):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db.execute(
        """
        INSERT INTO questions (
            question_id, raw_text, predicate, event_family, subject,
            deadline, resolution_rule, status, created_at, as_of_time
        ) VALUES ('q1', 'Will X happen?', 'military_action', 'conflict', 'X', ?, 'rule', 'open', ?, ?)
        """,
        [date(2026, 6, 1), now, now],
    )
    db.execute(
        """
        INSERT INTO feature_snapshots (
            snapshot_id, question_id, as_of_time, explicit_features, feature_schema_ver,
            builder_version, built_at
        ) VALUES ('s1', 'q1', ?, '{"f1": 0.3, "f2": 0.1}', 'v3', 'v3', ?)
        """,
        [now, now],
    )
    db.execute(
        """
        INSERT INTO predictions (
            prediction_id, question_id, snapshot_id,
            raw_prob, calibrated_prob, answer,
            model_id, predicted_at, as_of_time
        ) VALUES ('p1', 'q1', 's1', 0.8, 0.8, 'YES', 'xgb', ?, ?)
        """,
        [now, now],
    )


def test_cmd_label_uses_lakehouse_primary_and_updates_metrics(temp_lakehouse):
    db = temp_lakehouse
    _seed_prediction_graph(db)

    import main

    main.cmd_label(SimpleNamespace(prediction_id="p1", outcome="yes"))

    fs = db.execute("SELECT outcome, outcome_resolved_at FROM feature_snapshots WHERE snapshot_id='s1'").fetchone()
    assert fs[0] == 1
    assert fs[1] is not None

    p = db.execute("SELECT brier_component, was_correct FROM predictions WHERE prediction_id='p1'").fetchone()
    assert p[0] == pytest.approx((0.8 - 1.0) ** 2)
    assert p[1] is True


def test_load_training_data_from_lakehouse_reads_snapshots(temp_lakehouse):
    db = temp_lakehouse
    _seed_prediction_graph(db)
    db.execute(
        """
        INSERT INTO question_resolutions (
            resolution_id, question_id, outcome, resolved_at, deadline_was, resolver_source
        ) VALUES ('r1', 'q1', 1, ?, ?, 'user')
        """,
        [datetime(2026, 2, 1, tzinfo=timezone.utc), date(2026, 6, 1)],
    )
    from data_layer.resolution import resolve_outcome_by_prediction_id
    assert resolve_outcome_by_prediction_id("p1", 1) is True

    from model.trainer import load_training_data_from_lakehouse

    X, y = load_training_data_from_lakehouse()
    assert len(X) == 1
    assert y == [1]
    assert X[0]["f1"] == pytest.approx(0.3)


def test_backtest_loader_uses_lakehouse_not_legacy_json(temp_lakehouse, tmp_path, monkeypatch):
    db = temp_lakehouse
    _seed_prediction_graph(db)
    db.execute(
        """
        INSERT INTO question_resolutions (
            resolution_id, question_id, outcome, resolved_at, deadline_was, resolver_source
        ) VALUES ('r1', 'q1', 0, ?, ?, 'user')
        """,
        [datetime(2026, 2, 1, tzinfo=timezone.utc), date(2026, 6, 1)],
    )
    from data_layer.resolution import resolve_outcome_by_prediction_id
    assert resolve_outcome_by_prediction_id("p1", 0) is True

    # Add legacy JSON that would fail if read as primary.
    legacy_dir = tmp_path / "data" / "training"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "bad.json").write_text('{"outcome": 1, "features": {}, "timestamp": "2000-01-01"}')
    monkeypatch.chdir(tmp_path)

    from scripts.backtest import _load_examples

    examples = _load_examples()
    assert len(examples) == 1
    assert examples[0]["source"] == "lakehouse"


def test_legacy_label_fallback_is_explicitly_deprecated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pred_dir = Path("data/predictions")
    pred_dir.mkdir(parents=True)
    (pred_dir / "pred_abc.json").write_text(
        '{"question":"Q?","timestamp":"2026-01-01","features":{"x":1.0}}'
    )

    from data_layer.resolution import resolve_outcome_legacy_json

    ok = resolve_outcome_legacy_json("abc", 1)
    assert ok is True
    updated = (pred_dir / "pred_abc.json").read_text()
    assert '"deprecated": true' in updated

    train_file = Path("data/training/pred_abc.json")
    assert train_file.exists()
    assert '"legacy": true' in train_file.read_text()
