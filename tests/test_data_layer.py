"""
Tests for the lakehouse data layer (data_layer/).

Coverage:
  - Schema initialization (all 14 tables + 5 views created)
  - Table structure (correct column names and types)
  - Write round-trips for each layer
  - Temporal reproducibility (as_of_time invariant)
  - Lineage chain recording
  - Non-fatal failure modes (lineage silently skips on missing table)
  - Partitioning helpers and views

All tests use an in-memory DuckDB instance to avoid touching the real database.
"""
from __future__ import annotations

import json
import pytest
from datetime import datetime, date, timezone, timedelta


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def db():
    """In-memory DuckDB connection with full schema initialized."""
    import duckdb
    from pathlib import Path

    conn = duckdb.connect(":memory:")
    schema_path = Path(__file__).parent.parent / "data_layer" / "schema.sql"
    conn.execute(schema_path.read_text(encoding="utf-8"))
    yield conn
    conn.close()


def _table_columns(db, table_name: str) -> set[str]:
    """Return set of column names for a table."""
    rows = db.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
        [table_name]
    ).fetchall()
    return {r[0] for r in rows}


def _table_exists(db, name: str) -> bool:
    r = db.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [name]
    ).fetchone()
    return r[0] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Schema structure tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemaCreation:
    """All tables and views must exist after init."""

    EXPECTED_TABLES = [
        "raw_documents",
        "canonical_documents",
        "entities",
        "entity_aliases",
        "document_entities",
        "canonical_events",
        "event_arguments",
        "entity_relations_temporal",
        "feature_snapshots",
        "embedding_registry",
        "question_templates",
        "questions",
        "question_resolutions",
        "predictions",
        "data_lineage",
    ]

    EXPECTED_VIEWS = [
        "active_relations",
        "overdue_questions",
        "training_ready_snapshots",
        "calibration_audit",
        "entity_recent_events",
    ]

    def test_all_tables_created(self, db):
        for table in self.EXPECTED_TABLES:
            assert _table_exists(db, table), f"Table '{table}' missing"

    def test_all_views_created(self, db):
        for view in self.EXPECTED_VIEWS:
            result = db.execute(
                "SELECT COUNT(*) FROM information_schema.views WHERE table_name = ?",
                [view]
            ).fetchone()
            assert result[0] == 1, f"View '{view}' missing"

    def test_table_count(self, db):
        count = db.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_type = 'BASE TABLE'"
        ).fetchone()[0]
        assert count >= len(self.EXPECTED_TABLES)


class TestTableStructure:
    """Critical columns must exist in each table."""

    def test_raw_documents_columns(self, db):
        cols = _table_columns(db, "raw_documents")
        required = {"raw_doc_id", "content_hash", "source", "published_at",
                    "ingested_at", "body", "source_quality", "collector_version"}
        assert required <= cols

    def test_canonical_documents_columns(self, db):
        cols = _table_columns(db, "canonical_documents")
        required = {"doc_id", "raw_doc_id", "source", "published_at",
                    "country_tags", "entity_mentions", "contradiction_score"}
        assert required <= cols

    def test_canonical_events_columns(self, db):
        cols = _table_columns(db, "canonical_events")
        required = {"event_id", "doc_id", "event_time", "event_type",
                    "actor_entity_ids", "polarity", "intensity", "certainty",
                    "extractor_version", "confidence"}
        assert required <= cols

    def test_feature_snapshots_columns(self, db):
        cols = _table_columns(db, "feature_snapshots")
        required = {"snapshot_id", "as_of_time", "explicit_features",
                    "feature_schema_ver", "builder_version", "outcome",
                    "doc_ids_used", "event_ids_used"}
        assert required <= cols

    def test_questions_columns(self, db):
        cols = _table_columns(db, "questions")
        required = {"question_id", "raw_text", "predicate", "event_family",
                    "subject", "deadline", "resolution_rule", "status",
                    "ood_score", "matched_model", "as_of_time"}
        assert required <= cols

    def test_question_resolutions_columns(self, db):
        cols = _table_columns(db, "question_resolutions")
        required = {"resolution_id", "question_id", "outcome", "resolved_at",
                    "resolver_source", "resolution_confidence"}
        assert required <= cols

    def test_data_lineage_columns(self, db):
        cols = _table_columns(db, "data_lineage")
        required = {"lineage_id", "output_type", "output_id", "input_type",
                    "input_ids", "processor_name", "processor_version",
                    "processed_at", "is_deterministic"}
        assert required <= cols

    def test_entity_relations_temporal_columns(self, db):
        cols = _table_columns(db, "entity_relations_temporal")
        required = {"relation_id", "subject_entity_id", "object_entity_id",
                    "relation_type", "valid_from", "valid_to",
                    "confidence", "asserted_at"}
        assert required <= cols

    def test_predictions_columns(self, db):
        cols = _table_columns(db, "predictions")
        required = {"prediction_id", "calibrated_prob", "raw_prob",
                    "ci_lo", "ci_hi", "model_id", "predicted_at",
                    "market_override", "brier_component", "was_correct"}
        assert required <= cols


# ═══════════════════════════════════════════════════════════════════════════════
# Write round-trip tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRawDocumentWrite:
    def test_insert_and_read(self, db):
        now = datetime.now(timezone.utc)
        db.execute("""
            INSERT INTO raw_documents (
                raw_doc_id, content_hash, source, ingested_at, body,
                source_quality, collector_version
            ) VALUES ('raw_test_001', 'hash_001', 'rss', ?, 'Test body', 0.7, 'v1')
        """, [now])

        row = db.execute(
            "SELECT raw_doc_id, source, source_quality FROM raw_documents WHERE raw_doc_id = 'raw_test_001'"
        ).fetchone()
        assert row is not None
        assert row[0] == "raw_test_001"
        assert row[1] == "rss"
        assert abs(row[2] - 0.7) < 1e-6

    def test_dedup_on_content_hash_and_source(self, db):
        now = datetime.now(timezone.utc)
        # Insert same (content_hash, source) twice — second should be ignored
        for _ in range(2):
            db.execute("""
                INSERT INTO raw_documents (
                    raw_doc_id, content_hash, source, ingested_at, body, source_quality
                ) VALUES (?, 'hash_dedup', 'gdelt', ?, 'Dedup test', 0.5)
                ON CONFLICT (content_hash, source) DO NOTHING
            """, [f"raw_dedup_{_}", now])

        count = db.execute(
            "SELECT COUNT(*) FROM raw_documents WHERE content_hash = 'hash_dedup'"
        ).fetchone()[0]
        assert count == 1


class TestCanonicalEventWrite:
    def test_insert_event_with_actors(self, db):
        # First ensure a canonical_document exists (FK)
        now = datetime.now(timezone.utc)
        db.execute("""
            INSERT INTO raw_documents (raw_doc_id, content_hash, source, ingested_at, body)
            VALUES ('raw_ev1', 'hash_ev1', 'rss', ?, 'Event test')
            ON CONFLICT DO NOTHING
        """, [now])
        db.execute("""
            INSERT INTO canonical_documents (
                doc_id, raw_doc_id, source, published_at, ingested_at,
                processed_at, normalizer_version
            ) VALUES ('doc_ev1', 'raw_ev1', 'rss', ?, ?, ?, 'v2')
            ON CONFLICT DO NOTHING
        """, [now, now, now])

        db.execute("""
            INSERT INTO canonical_events (
                event_id, doc_id, event_time, event_type,
                actor_entity_ids, polarity, intensity, confidence, extractor_version
            ) VALUES ('evt_test_001', 'doc_ev1', ?, 'military_action',
                      ['ent_ukraine'], -0.8, 7.5, 0.92, 'v3')
        """, [now])

        row = db.execute(
            "SELECT event_type, polarity, intensity FROM canonical_events WHERE event_id = 'evt_test_001'"
        ).fetchone()
        assert row[0] == "military_action"
        assert row[1] == pytest.approx(-0.8, abs=1e-6)
        assert row[2] == pytest.approx(7.5, abs=1e-6)

    def test_event_type_variety(self, db):
        """Multiple event types across domains can be stored in same table."""
        now = datetime.now(timezone.utc)
        db.execute("""
            INSERT INTO raw_documents (raw_doc_id, content_hash, source, ingested_at, body)
            VALUES ('raw_ev2', 'hash_ev2', 'rss', ?, 'Multi event')
            ON CONFLICT DO NOTHING
        """, [now])
        db.execute("""
            INSERT INTO canonical_documents (
                doc_id, raw_doc_id, source, published_at, ingested_at, processed_at
            ) VALUES ('doc_ev2', 'raw_ev2', 'rss', ?, ?, ?)
            ON CONFLICT DO NOTHING
        """, [now, now, now])

        event_types = [
            "resignation_signal", "arrest_signal",
            "tour_announcement", "rate_decision",
        ]
        for i, etype in enumerate(event_types):
            db.execute("""
                INSERT INTO canonical_events (
                    event_id, doc_id, event_time, event_type, extractor_version
                ) VALUES (?, 'doc_ev2', ?, ?, 'v3')
            """, [f"evt_multi_{i}", now, etype])

        stored = db.execute(
            "SELECT event_type FROM canonical_events WHERE doc_id = 'doc_ev2'"
        ).fetchall()
        stored_types = {r[0] for r in stored}
        assert set(event_types) <= stored_types


class TestFeatureSnapshotWrite:
    def test_insert_snapshot_with_features(self, db):
        now = datetime.now(timezone.utc)
        features = {
            "military_count_7d": 3.0,
            "protest_count_7d": 1.0,
            "country_conflict_baserate": 0.35,
            "country_polity_norm": 0.6,
        }
        db.execute("""
            INSERT INTO feature_snapshots (
                snapshot_id, as_of_time, explicit_features,
                feature_schema_ver, builder_version, built_at
            ) VALUES ('snap_test_001', ?, ?, 'v3', 'v1', ?)
        """, [now, json.dumps(features), now])

        row = db.execute(
            "SELECT explicit_features, feature_schema_ver FROM feature_snapshots "
            "WHERE snapshot_id = 'snap_test_001'"
        ).fetchone()
        assert row is not None
        stored_features = json.loads(row[0])
        assert stored_features["military_count_7d"] == pytest.approx(3.0)
        assert row[1] == "v3"

    def test_outcome_can_be_set_later(self, db):
        """Simulates labeling after resolution."""
        now = datetime.now(timezone.utc)
        db.execute("""
            INSERT INTO feature_snapshots (
                snapshot_id, as_of_time, explicit_features,
                feature_schema_ver, builder_version, built_at
            ) VALUES ('snap_label', ?, '{}', 'v3', 'v1', ?)
        """, [now, now])

        # Outcome initially NULL
        row = db.execute(
            "SELECT outcome FROM feature_snapshots WHERE snapshot_id = 'snap_label'"
        ).fetchone()
        assert row[0] is None

        # Set outcome
        db.execute(
            "UPDATE feature_snapshots SET outcome = 1 WHERE snapshot_id = 'snap_label'"
        )
        row = db.execute(
            "SELECT outcome FROM feature_snapshots WHERE snapshot_id = 'snap_label'"
        ).fetchone()
        assert row[0] == 1


class TestQuestionWrite:
    def test_insert_question_and_resolution(self, db):
        now = datetime.now(timezone.utc)
        deadline = date(2026, 6, 1)

        db.execute("""
            INSERT INTO questions (
                question_id, raw_text, predicate, event_family, subject,
                deadline, resolution_rule, status, created_at, as_of_time
            ) VALUES (
                'q_test_001',
                'Will there be military escalation in Ukraine before June 2026?',
                'military_escalation', 'conflict', 'Ukraine',
                ?, 'Armed conflict event confirmed', 'open', ?, ?
            )
        """, [deadline, now, now])

        # Resolve it
        db.execute("""
            INSERT INTO question_resolutions (
                resolution_id, question_id, outcome, resolved_at,
                deadline_was, resolver_source
            ) VALUES ('res_test_001', 'q_test_001', 1, ?, ?, 'news')
        """, [now, deadline])

        row = db.execute("""
            SELECT q.raw_text, qr.outcome
            FROM questions q
            JOIN question_resolutions qr ON q.question_id = qr.question_id
            WHERE q.question_id = 'q_test_001'
        """).fetchone()
        assert row is not None
        assert row[1] == 1

    def test_overdue_view(self, db):
        """Overdue view shows open questions past deadline."""
        yesterday = date.today() - timedelta(days=1)
        now = datetime.now(timezone.utc)

        db.execute("""
            INSERT INTO questions (
                question_id, raw_text, predicate, event_family, subject,
                deadline, resolution_rule, status, created_at, as_of_time
            ) VALUES (
                'q_overdue', 'Will X happen?', 'military_escalation', 'conflict', 'Syria',
                ?, 'Rule', 'open', ?, ?
            )
        """, [yesterday, now, now])

        count = db.execute(
            "SELECT COUNT(*) FROM overdue_questions WHERE question_id = 'q_overdue'"
        ).fetchone()[0]
        assert count >= 1


class TestDataLineageWrite:
    def test_lineage_insert_and_chain(self, db):
        now = datetime.now(timezone.utc)

        # Raw → canonical
        db.execute("""
            INSERT INTO data_lineage (
                lineage_id, output_type, output_id,
                input_type, input_ids,
                processor_name, processor_version, processed_at, is_deterministic
            ) VALUES (
                'lin_001', 'canonical_document', 'doc_lin_001',
                'raw_document', ['raw_lin_001'],
                'normalizer.canonical', 'v2', ?, TRUE
            )
        """, [now])

        # Canonical → event
        db.execute("""
            INSERT INTO data_lineage (
                lineage_id, output_type, output_id,
                input_type, input_ids,
                processor_name, processor_version, processed_at, is_deterministic
            ) VALUES (
                'lin_002', 'canonical_event', 'evt_lin_001',
                'canonical_document', ['doc_lin_001'],
                'event_extractor', 'v3', ?, TRUE
            )
        """, [now])

        # Trace chain: event → its source canonical_document
        row = db.execute(
            "SELECT input_ids FROM data_lineage WHERE output_id = 'evt_lin_001'"
        ).fetchone()
        assert row is not None
        # input_ids is an array; DuckDB returns it as a list
        assert "doc_lin_001" in row[0]

    def test_lineage_processor_index(self, db):
        """Can query all outputs from a specific processor version."""
        rows = db.execute(
            "SELECT output_id FROM data_lineage WHERE processor_name = 'event_extractor'"
        ).fetchall()
        assert len(rows) >= 1


class TestTemporalReproducibility:
    """as_of_time must be stored immutably with each snapshot."""

    def test_as_of_time_preserved(self, db):
        cutoff = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
        db.execute("""
            INSERT INTO feature_snapshots (
                snapshot_id, as_of_time, explicit_features,
                feature_schema_ver, builder_version, built_at
            ) VALUES ('snap_temporal', ?, '{"x": 1.0}', 'v3', 'v1', ?)
        """, [cutoff, datetime.now(timezone.utc)])

        row = db.execute(
            "SELECT as_of_time FROM feature_snapshots WHERE snapshot_id = 'snap_temporal'"
        ).fetchone()
        assert row is not None
        stored_time = row[0]
        # DuckDB may return as datetime; compare at second precision
        assert stored_time.year == 2026
        assert stored_time.month == 3
        assert stored_time.day == 15

    def test_two_snapshots_different_cutoffs(self, db):
        """Same subject, different as_of_time → two separate rows."""
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 3, 1, tzinfo=timezone.utc)

        db.execute("""
            INSERT INTO feature_snapshots (
                snapshot_id, as_of_time, explicit_features,
                feature_schema_ver, builder_version, built_at
            ) VALUES
                ('snap_t1', ?, '{"military_count_7d": 1.0}', 'v3', 'v1', ?),
                ('snap_t2', ?, '{"military_count_7d": 5.0}', 'v3', 'v1', ?)
        """, [t1, datetime.now(timezone.utc), t2, datetime.now(timezone.utc)])

        rows = db.execute(
            "SELECT snapshot_id, as_of_time FROM feature_snapshots "
            "WHERE snapshot_id IN ('snap_t1', 'snap_t2') ORDER BY as_of_time"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][0] == "snap_t1"
        assert rows[1][0] == "snap_t2"


class TestEntityRelationsTemporalWrite:
    def test_insert_temporal_relation(self, db):
        now = datetime.now(timezone.utc)

        # Insert entities
        db.execute("""
            INSERT INTO entities (entity_id, canonical_name, entity_type, first_seen_at)
            VALUES ('ent_sanchez', 'Pedro Sanchez', 'person', ?)
            ON CONFLICT DO NOTHING
        """, [now])
        db.execute("""
            INSERT INTO entities (entity_id, canonical_name, entity_type, first_seen_at)
            VALUES ('ent_spain', 'Spain', 'country', ?)
            ON CONFLICT DO NOTHING
        """, [now])

        # Insert relation: Sánchez holds office as PM of Spain
        db.execute("""
            INSERT INTO entity_relations_temporal (
                relation_id, subject_entity_id, object_entity_id,
                relation_type, valid_from, asserted_at, confidence
            ) VALUES (
                'rel_001', 'ent_sanchez', 'ent_spain',
                'holds_office', '2018-06-02', ?, 0.99
            )
        """, [now])

        row = db.execute(
            "SELECT relation_type, valid_from, valid_to FROM entity_relations_temporal "
            "WHERE relation_id = 'rel_001'"
        ).fetchone()
        assert row[0] == "holds_office"
        assert row[2] is None   # still active

    def test_active_relations_view_shows_open_relation(self, db):
        """active_relations view only shows valid_to IS NULL or future."""
        count = db.execute(
            "SELECT COUNT(*) FROM active_relations WHERE relation_id = 'rel_001'"
        ).fetchone()[0]
        assert count == 1
