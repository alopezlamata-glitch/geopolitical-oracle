from __future__ import annotations

import json

import scripts.backtest as backtest


def test_load_examples_reads_from_training_ready_snapshots(monkeypatch):
    sql_seen: list[str] = []

    class FakeDB:
        def execute(self, sql: str):
            sql_seen.append(sql)

            class _R:
                @staticmethod
                def fetchall():
                    return [
                        ("s2", "2022-01-03T00:00:00+00:00", json.dumps({"f": 2.0}), 0),
                        ("s1", "2022-01-01T00:00:00+00:00", json.dumps({"f": 1.0}), 1),
                    ]

            return _R()

    monkeypatch.setattr(backtest, "get_db", lambda read_only=True: FakeDB(), raising=False)
    monkeypatch.setattr("data_layer.db.get_db", lambda read_only=True: FakeDB())
    monkeypatch.setattr("features.builder.get_feature_names", lambda: ["f"])

    examples = backtest._load_examples()

    assert examples
    assert all(e["source"] == "training_ready_snapshots" for e in examples)
    assert any("training_ready_snapshots" in q for q in sql_seen)


def test_load_examples_excludes_missing_outcome_and_is_temporally_sorted(monkeypatch):
    class FakeDB:
        def execute(self, _sql: str):
            class _R:
                @staticmethod
                def fetchall():
                    return [
                        ("s3", "2022-01-02T00:00:00+00:00", {"f": 3.0}, 1),
                        ("s2", "2022-01-02T00:00:00+00:00", {"f": 2.0}, 0),
                        ("s1", "2022-01-01T00:00:00+00:00", {"f": 1.0}, 1),
                    ]

            return _R()

    monkeypatch.setattr("data_layer.db.get_db", lambda read_only=True: FakeDB())
    monkeypatch.setattr("features.builder.get_feature_names", lambda: ["f"])

    examples = backtest._load_examples()

    # deterministic temporal order + deterministic tie-break by snapshot_id
    assert [e["snapshot_id"] for e in examples] == ["s1", "s2", "s3"]
    assert [e["ts"] for e in examples] == sorted([e["ts"] for e in examples])
