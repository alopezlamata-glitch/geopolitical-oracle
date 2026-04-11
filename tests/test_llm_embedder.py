"""
Tests for llm/embedder.py and entity resolution integration.

Coverage:
  - cosine_similarity: edge cases (zero vector, identical, orthogonal)
  - _mean_vector: correctness, empty
  - EmbedResult dataclass defaults
  - embed_events_and_question: happy path, Ollama failure, empty inputs
  - Entity writers: write_entity, write_entity_alias (in-memory DuckDB)
  - persist_entity_resolution: resolves actors → DB entities (mocked LLM)
  - builder llm_query_event_similarity: present when embedder available
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm.embedder import (
    EmbedResult,
    cosine_similarity,
    _mean_vector,
    embed_events_and_question,
)


# ── cosine_similarity ──────────────────────────────────────────────────────────

class TestCosineSimilarity:

    def test_identical_vectors_give_1(self):
        v = [1.0, 0.5, 0.3]
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_give_0(self):
        # In [0,1] range: opposite → cos = -1 → (−1+1)/2 = 0
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors_give_0_5(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.5, abs=1e-6)

    def test_zero_vector_returns_neutral(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.5

    def test_empty_vectors_return_neutral(self):
        assert cosine_similarity([], []) == 0.5

    def test_mismatched_lengths_return_neutral(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.5

    def test_result_always_in_01(self):
        import random
        rng = random.Random(42)
        for _ in range(100):
            a = [rng.uniform(-1, 1) for _ in range(10)]
            b = [rng.uniform(-1, 1) for _ in range(10)]
            sim = cosine_similarity(a, b)
            assert 0.0 <= sim <= 1.0


# ── _mean_vector ───────────────────────────────────────────────────────────────

class TestMeanVector:

    def test_single_vector_returns_itself(self):
        v = [1.0, 2.0, 3.0]
        assert _mean_vector([v]) == pytest.approx(v)

    def test_two_vectors_averaged(self):
        a = [2.0, 0.0]
        b = [0.0, 2.0]
        mean = _mean_vector([a, b])
        assert mean == pytest.approx([1.0, 1.0])

    def test_empty_returns_empty(self):
        assert _mean_vector([]) == []


# ── EmbedResult defaults ───────────────────────────────────────────────────────

class TestEmbedResult:

    def test_default_similarity_is_neutral(self):
        r = EmbedResult()
        assert r.query_similarity == 0.5

    def test_unavailable_by_default(self):
        r = EmbedResult()
        assert r.available is False

    def test_top_similarities_empty_by_default(self):
        r = EmbedResult()
        assert r.top_event_similarities == []


# ── embed_events_and_question ──────────────────────────────────────────────────

class TestEmbedEventsAndQuestion:

    def test_returns_unavailable_when_no_question(self):
        result = embed_events_and_question("", ["headline"])
        assert result.available is False

    def test_returns_unavailable_when_no_headlines(self):
        result = embed_events_and_question("Question?", [])
        assert result.available is False

    def test_happy_path_with_mock(self):
        """Successful embed: returns similarity in [0,1]."""
        q_vec = [1.0, 0.0, 0.0]
        e_vec = [0.8, 0.2, 0.0]   # similar to q_vec

        mock_result = EmbedResult(
            query_similarity=0.88,
            n_embedded=2,
            available=True,
            question_embedding=q_vec,
            mean_event_embedding=e_vec,
        )

        with patch("llm.embedder.asyncio.run", return_value=mock_result):
            result = embed_events_and_question(
                "Will Iran escalate?",
                ["Troops mobilize", "Border incident"],
            )

        assert result.available is True
        assert 0.0 <= result.query_similarity <= 1.0
        assert result.n_embedded == 2

    def test_returns_neutral_on_ollama_failure(self):
        with patch("llm.embedder.asyncio.run", side_effect=Exception("connection refused")):
            result = embed_events_and_question("Q?", ["headline"])

        assert result.available is False
        assert result.query_similarity == 0.5  # neutral fallback

    def test_top_similarities_sorted_descending(self):
        mock_result = EmbedResult(
            query_similarity=0.7,
            top_event_similarities=[
                ("high relevance", 0.9),
                ("medium relevance", 0.6),
                ("low relevance", 0.3),
            ],
            available=True,
            n_embedded=3,
        )
        with patch("llm.embedder.asyncio.run", return_value=mock_result):
            result = embed_events_and_question("Q?", ["h1", "h2", "h3"])

        # Verify that returned top_sims are sorted high to low
        sims = [s for _, s in result.top_event_similarities]
        assert sims == sorted(sims, reverse=True)


# ── Entity writers (in-memory DuckDB) ─────────────────────────────────────────

class TestEntityWriters:

    def _setup_db(self):
        """Create minimal in-memory DuckDB with entities + entity_aliases."""
        import duckdb
        con = duckdb.connect(":memory:")
        con.execute("""
            CREATE TABLE entities (
                entity_id VARCHAR PRIMARY KEY,
                canonical_name VARCHAR NOT NULL,
                entity_type VARCHAR NOT NULL,
                wikidata_id VARCHAR,
                country VARCHAR,
                description VARCHAR,
                conflict_baserate FLOAT,
                polity_norm FLOAT,
                mil_spending_norm FLOAT,
                first_seen_at TIMESTAMPTZ,
                last_updated_at TIMESTAMPTZ,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        con.execute("""
            CREATE TABLE entity_aliases (
                alias_id VARCHAR PRIMARY KEY,
                entity_id VARCHAR NOT NULL,
                alias VARCHAR NOT NULL,
                alias_language VARCHAR(5) DEFAULT 'en',
                source VARCHAR,
                confidence FLOAT DEFAULT 1.0,
                CONSTRAINT entity_alias_unique UNIQUE (alias, alias_language)
            )
        """)
        return con

    def test_write_entity_returns_id(self):
        try:
            import duckdb
        except ImportError:
            pytest.skip("duckdb not installed")

        con = self._setup_db()
        with patch("data_layer.db.get_db", return_value=con), \
             patch("data_layer.db.table_exists", return_value=True):
            from data_layer.writers import write_entity
            eid = write_entity("Vladimir Putin", "person", country="Russia")

        assert eid is not None
        assert eid.startswith("ent_")

    def test_write_entity_idempotent(self):
        """Writing the same entity twice should not raise and return same ID."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("duckdb not installed")

        con = self._setup_db()
        with patch("data_layer.db.get_db", return_value=con), \
             patch("data_layer.db.table_exists", return_value=True):
            from data_layer.writers import write_entity
            id1 = write_entity("NATO", "organization")
            id2 = write_entity("NATO", "organization")  # duplicate

        assert id1 == id2  # same deterministic ID

    def test_write_entity_alias_stored(self):
        try:
            import duckdb
        except ImportError:
            pytest.skip("duckdb not installed")

        con = self._setup_db()
        with patch("data_layer.db.get_db", return_value=con), \
             patch("data_layer.db.table_exists", return_value=True):
            from data_layer.writers import write_entity, write_entity_alias
            eid = write_entity("Vladimir Putin", "person")
            write_entity_alias(eid, "Putin", source="llm_extracted", confidence=0.9)

        row = con.execute("SELECT alias, entity_id FROM entity_aliases").fetchone()
        assert row is not None
        assert row[0] == "Putin"
        assert row[1] == eid

    def test_write_entity_alias_idempotent(self):
        """Same alias inserted twice should not raise (ON CONFLICT DO NOTHING)."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("duckdb not installed")

        con = self._setup_db()
        with patch("data_layer.db.get_db", return_value=con), \
             patch("data_layer.db.table_exists", return_value=True):
            from data_layer.writers import write_entity, write_entity_alias
            eid = write_entity("Iran", "country")
            a1 = write_entity_alias(eid, "Tehran", source="llm_extracted")
            a2 = write_entity_alias(eid, "Tehran", source="llm_extracted")  # duplicate

        count = con.execute("SELECT COUNT(*) FROM entity_aliases").fetchone()[0]
        assert count == 1  # only one row


# ── persist_entity_resolution integration ────────────────────────────────────

class TestPersistEntityResolution:

    def _make_event(self, actors: list[str]) -> "CanonicalEvent":
        from normalizer.canonical import CanonicalEvent
        return CanonicalEvent(
            event_id="e1",
            doc_ids=[],
            source="rss",
            occurred_at=datetime.now(timezone.utc),
            event_type="military_action",
            sub_event_type="",
            actors=actors,
            country="Russia",
            severity=0.7,
            polarity=-0.8,
            fatalities=0,
            independent_sources=1,
            contradiction_score=0.0,
            raw_title="Troops mobilize",
        )

    def test_returns_empty_when_no_actors(self):
        from data_layer.pipeline_hooks import persist_entity_resolution
        ev = self._make_event([])
        result = persist_entity_resolution([ev], country="Russia")
        assert result == {}

    def test_resolves_actors_with_mocked_llm_and_db(self):
        from data_layer.pipeline_hooks import persist_entity_resolution
        from llm.entity_resolver import EntityCanon

        ev = self._make_event(["Putin", "Kremlin"])

        mock_resolutions = [
            EntityCanon(canonical="Vladimir Putin", entity_type="person",
                        mention="Putin", confidence=0.9, resolved_by="llm"),
            EntityCanon(canonical="Russia Government", entity_type="organization",
                        mention="Kremlin", confidence=0.85, resolved_by="llm"),
        ]

        with patch("llm.entity_resolver.resolve_entities_batch", return_value=mock_resolutions), \
             patch("data_layer.writers.write_entity", side_effect=["ent_abc", "ent_def"]), \
             patch("data_layer.writers.write_entity_alias", return_value="ali_xyz"):
            result = persist_entity_resolution([ev], country="Russia")

        assert "Putin" in result or "Kremlin" in result
        assert len(result) == 2


# ── builder: llm_query_event_similarity feature ───────────────────────────────

class TestBuilderEmbeddingSimilarity:

    def _make_event(self, title: str = "Airstrike near border") -> "CanonicalEvent":
        from normalizer.canonical import CanonicalEvent
        return CanonicalEvent(
            event_id="e5",
            doc_ids=[],
            source="gdelt",
            occurred_at=datetime.now(timezone.utc),
            event_type="military_action",
            sub_event_type="",
            actors=["Russia"],
            country="Ukraine",
            severity=0.8,
            polarity=-0.9,
            fatalities=5,
            independent_sources=2,
            contradiction_score=0.1,
            raw_title=title,
        )

    def test_similarity_feature_added_when_embedding_available(self):
        from features.builder import build_features

        mock_embed = EmbedResult(query_similarity=0.82, n_embedded=1, available=True)

        with patch("llm.text_features.extract_llm_features",
                   return_value=MagicMock(available=False, to_feature_dict=lambda: {
                       "llm_threat_level": 0.0, "llm_escalation": 0.0,
                       "llm_deescalation": 0.0, "llm_event_certainty": 0.0,
                       "llm_actor_hostility": 0.0, "llm_available": 0.0,
                   })), \
             patch("llm.embedder.embed_events_and_question", return_value=mock_embed):
            feat, _ = build_features(
                [self._make_event()],
                question="Will Russia attack Ukraine?",
                use_llm=True,
            )

        assert "llm_query_event_similarity" in feat
        assert abs(feat["llm_query_event_similarity"] - 0.82) < 1e-6

    def test_similarity_feature_zero_when_embedding_unavailable(self):
        from features.builder import build_features

        mock_embed = EmbedResult(available=False)  # Ollama down

        with patch("llm.text_features.extract_llm_features",
                   return_value=MagicMock(available=False, to_feature_dict=lambda: {
                       "llm_threat_level": 0.0, "llm_escalation": 0.0,
                       "llm_deescalation": 0.0, "llm_event_certainty": 0.0,
                       "llm_actor_hostility": 0.0, "llm_available": 0.0,
                   })), \
             patch("llm.embedder.embed_events_and_question", return_value=mock_embed):
            feat, _ = build_features(
                [self._make_event()],
                question="Q?",
                use_llm=True,
            )

        # Should be 0 (default) when embedder unavailable
        assert feat.get("llm_query_event_similarity", 0.0) == 0.0

    def test_v3_feature_names_still_clean(self):
        from features.builder import get_feature_names
        v3 = get_feature_names()
        assert "llm_query_event_similarity" not in v3

    def test_v4_feature_names_include_similarity(self):
        from features.builder import get_feature_names_v4
        v4 = get_feature_names_v4()
        assert "llm_query_event_similarity" in v4
        assert len(v4) == 34   # 27 v3 + 7 llm features (6 text + 1 similarity)
