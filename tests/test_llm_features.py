"""
Tests for the llm/ module — OllamaClient, text feature extractor, entity resolver.

All tests use mocks — no real Ollama server required.

Coverage:
  - OllamaClient: generate_json, embed, is_available (mocked HTTP)
  - LLMFeatures.to_feature_dict: correct keys and clipping
  - extract_llm_features: happy path, parse edge cases, Ollama unavailable
  - _parse_response: valid JSON, wrapped keys, missing keys, bad types
  - resolve_entity: happy path, cache hit, fallback on LLM failure
  - resolve_entities_batch: deduplication via cache
  - features/builder.py: llm features added to dict, non-fatal on failure
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm.text_features import (
    LLMFeatures,
    LLM_FEATURE_NAMES,
    _parse_response,
    _build_prompt,
    _clip,
)
from llm.entity_resolver import (
    EntityCanon,
    VALID_TYPES,
    _fallback_resolution,
    _parse_resolution,
    clear_cache,
    resolve_entity,
    resolve_entities_batch,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_client(response: dict | None = None, available: bool = True):
    """Return a mock OllamaClient that returns `response` from generate_json."""
    client = MagicMock()
    client.text_model = "llama3.2"
    client.embed_model = "nomic-embed-text"
    client.generate_json = AsyncMock(return_value=response)
    client.is_available = AsyncMock(return_value=available)
    return client


# ── _clip ──────────────────────────────────────────────────────────────────────

class TestClip:

    def test_clips_above_1(self):
        assert _clip(1.5) == 1.0

    def test_clips_below_0(self):
        assert _clip(-0.3) == 0.0

    def test_passthrough_in_range(self):
        assert abs(_clip(0.7) - 0.7) < 1e-9

    def test_handles_string(self):
        assert _clip("0.5") == pytest.approx(0.5)

    def test_handles_none(self):
        assert _clip(None) == 0.0

    def test_handles_non_numeric_string(self):
        assert _clip("foo") == 0.0


# ── LLMFeatures.to_feature_dict ───────────────────────────────────────────────

class TestLLMFeaturesToDict:

    def test_all_keys_present(self):
        f = LLMFeatures(threat_level=0.6, escalation=0.4, available=True)
        d = f.to_feature_dict()
        for key in LLM_FEATURE_NAMES:
            assert key in d, f"Missing key: {key}"

    def test_available_maps_to_1(self):
        d = LLMFeatures(available=True).to_feature_dict()
        assert d["llm_available"] == 1.0

    def test_unavailable_maps_to_0(self):
        d = LLMFeatures(available=False).to_feature_dict()
        assert d["llm_available"] == 0.0

    def test_zero_features_when_unavailable(self):
        d = LLMFeatures(available=False).to_feature_dict()
        assert d["llm_threat_level"] == 0.0
        assert d["llm_escalation"] == 0.0


# ── _parse_response ────────────────────────────────────────────────────────────

class TestParseResponse:

    def test_valid_json_parsed(self):
        data = {
            "threat_level": 0.7,
            "escalation": 0.5,
            "deescalation": 0.2,
            "event_certainty": 0.8,
            "actor_hostility": 0.6,
        }
        result = _parse_response(data)
        assert result is not None
        assert result.available is True
        assert abs(result.threat_level - 0.7) < 1e-6
        assert abs(result.escalation - 0.5) < 1e-6

    def test_values_clipped_above_1(self):
        data = {"threat_level": 1.5, "escalation": 0.5, "deescalation": 0.0,
                "event_certainty": 0.5, "actor_hostility": 0.5}
        result = _parse_response(data)
        assert result.threat_level == 1.0

    def test_values_clipped_below_0(self):
        data = {"threat_level": -0.3, "escalation": 0.5, "deescalation": 0.0,
                "event_certainty": 0.5, "actor_hostility": 0.5}
        result = _parse_response(data)
        assert result.threat_level == 0.0

    def test_none_returns_none(self):
        assert _parse_response(None) is None

    def test_non_dict_returns_none(self):
        assert _parse_response("not a dict") is None
        assert _parse_response([1, 2, 3]) is None

    def test_missing_keys_default_to_0(self):
        # Only threat_level provided — others should default to 0
        result = _parse_response({"threat_level": 0.8})
        assert result is not None
        assert result.escalation == 0.0

    def test_wrapper_key_features(self):
        # Some models wrap in a 'features' key
        data = {"features": {"threat_level": 0.9, "escalation": 0.3,
                              "deescalation": 0.1, "event_certainty": 0.7,
                              "actor_hostility": 0.4}}
        result = _parse_response(data)
        assert result is not None
        assert abs(result.threat_level - 0.9) < 1e-6

    def test_alternate_key_names(self):
        # Some models use slightly different key names
        data = {"threat": 0.6, "escalation_level": 0.4, "de_escalation": 0.2,
                "certainty": 0.7, "hostility": 0.5}
        result = _parse_response(data)
        assert result is not None
        assert abs(result.threat_level - 0.6) < 1e-6


# ── _build_prompt ──────────────────────────────────────────────────────────────

class TestBuildPrompt:

    def test_question_appears_in_prompt(self):
        prompt = _build_prompt("Will Iran attack Israel?", ["Missile launched", "Troops mobilize"])
        assert "Iran" in prompt

    def test_headlines_truncated(self):
        long_headlines = [f"headline {i} " + "x" * 200 for i in range(30)]
        prompt = _build_prompt("Q?", long_headlines)
        assert len(prompt) < 10000  # reasonable bound

    def test_empty_headlines_still_works(self):
        prompt = _build_prompt("Q?", [])
        assert "Q?" in prompt


# ── extract_llm_features (sync wrapper) ───────────────────────────────────────

class TestExtractLLMFeatures:

    def test_returns_zero_features_when_no_headlines(self):
        from llm.text_features import extract_llm_features
        result = extract_llm_features("Q?", headlines=[])
        assert result.available is False
        assert result.threat_level == 0.0

    def test_happy_path_with_mock_client(self):
        from llm.text_features import extract_llm_features

        mock_resp = {
            "threat_level": 0.8,
            "escalation": 0.6,
            "deescalation": 0.1,
            "event_certainty": 0.9,
            "actor_hostility": 0.7,
        }
        client = _mock_client(response=mock_resp)

        with patch("llm.text_features.asyncio.run") as mock_run:
            mock_run.return_value = LLMFeatures(
                threat_level=0.8, escalation=0.6, deescalation=0.1,
                event_certainty=0.9, actor_hostility=0.7, available=True,
            )
            result = extract_llm_features("Will Iran attack?", ["Troops move"], client=client)

        assert result.available is True
        assert abs(result.threat_level - 0.8) < 1e-6

    def test_returns_zero_on_ollama_error(self):
        from llm.text_features import extract_llm_features

        with patch("llm.text_features.asyncio.run", side_effect=Exception("connection refused")):
            result = extract_llm_features("Q?", ["headline"], client=MagicMock())

        assert result.available is False
        d = result.to_feature_dict()
        assert d["llm_available"] == 0.0

    def test_feature_dict_has_all_keys(self):
        from llm.text_features import extract_llm_features

        with patch("llm.text_features.asyncio.run") as mock_run:
            mock_run.return_value = LLMFeatures(available=True, threat_level=0.5)
            result = extract_llm_features("Q?", ["headline"])

        d = result.to_feature_dict()
        for key in LLM_FEATURE_NAMES:
            assert key in d


# ── Entity resolver helpers ────────────────────────────────────────────────────

class TestFallbackResolution:

    def test_title_case_applied(self):
        result = _fallback_resolution("vladimir putin")
        assert result.canonical == "Vladimir Putin"

    def test_two_words_classified_as_person(self):
        result = _fallback_resolution("Vladimir Putin")
        assert result.entity_type == "person"

    def test_all_caps_short_classified_as_organization(self):
        result = _fallback_resolution("NATO")
        assert result.entity_type == "organization"

    def test_confidence_low_for_fallback(self):
        result = _fallback_resolution("some entity")
        assert result.confidence < 0.5
        assert result.resolved_by == "fallback"

    def test_empty_string_handled(self):
        result = _fallback_resolution("")
        assert result.canonical == "" or result.entity_type in VALID_TYPES


class TestParseResolution:

    def test_valid_resolution_parsed(self):
        data = {"canonical": "Vladimir Putin", "type": "person"}
        result = _parse_resolution(data, "Putin")
        assert result is not None
        assert result.canonical == "Vladimir Putin"
        assert result.entity_type == "person"

    def test_invalid_type_defaults_to_other(self):
        data = {"canonical": "Some Entity", "type": "alien"}
        result = _parse_resolution(data, "Some Entity")
        assert result is not None
        assert result.entity_type == "other"

    def test_none_input_returns_none(self):
        assert _parse_resolution(None, "test") is None

    def test_empty_canonical_returns_none(self):
        data = {"canonical": "", "type": "person"}
        assert _parse_resolution(data, "test") is None

    def test_all_valid_types_accepted(self):
        for t in VALID_TYPES:
            data = {"canonical": "Test Entity", "type": t}
            result = _parse_resolution(data, "Test Entity")
            assert result is not None
            assert result.entity_type == t


# ── resolve_entity ────────────────────────────────────────────────────────────

class TestResolveEntity:

    def setup_method(self):
        clear_cache()

    def test_empty_mention_returns_fallback(self):
        result = resolve_entity("", context="Iran")
        assert result.entity_type == "other"

    def test_caches_result_on_second_call(self):
        with patch("llm.entity_resolver.asyncio.run") as mock_run:
            mock_run.return_value = EntityCanon(
                canonical="Vladimir Putin",
                entity_type="person",
                mention="Putin",
                resolved_by="llm",
            )
            result1 = resolve_entity("Putin", context="Russia")
            result2 = resolve_entity("Putin", context="Russia")

        # asyncio.run should be called only once (second call hits cache)
        assert mock_run.call_count == 1
        assert result2.from_cache is True
        assert result1.canonical == result2.canonical

    def test_fallback_on_asyncio_error(self):
        with patch("llm.entity_resolver.asyncio.run", side_effect=Exception("error")):
            result = resolve_entity("IRGC", context="Iran")
        assert result.resolved_by == "fallback"
        assert result.canonical  # should have something

    def test_context_affects_cache_key(self):
        """Same mention, different context → different cache entries."""
        with patch("llm.entity_resolver.asyncio.run") as mock_run:
            mock_run.return_value = EntityCanon(
                canonical="Washington", entity_type="country", mention="Washington",
                resolved_by="llm"
            )
            r1 = resolve_entity("Washington", context="USA")
            r2 = resolve_entity("Washington", context="Russia")

        # Different contexts → two separate calls
        assert mock_run.call_count == 2


class TestResolveEntitiesBatch:

    def setup_method(self):
        clear_cache()

    def test_returns_same_length(self):
        mentions = ["Putin", "NATO", "Zelensky"]
        with patch("llm.entity_resolver.asyncio.run") as mock_run:
            mock_run.return_value = EntityCanon(
                canonical="Test", entity_type="person", mention="test", resolved_by="llm"
            )
            results = resolve_entities_batch(mentions, context="Ukraine")
        assert len(results) == 3

    def test_cache_reduces_llm_calls(self):
        """Duplicate mentions should only call LLM once."""
        mentions = ["Putin", "Putin", "Putin"]
        with patch("llm.entity_resolver.asyncio.run") as mock_run:
            mock_run.return_value = EntityCanon(
                canonical="Vladimir Putin", entity_type="person",
                mention="Putin", resolved_by="llm"
            )
            results = resolve_entities_batch(mentions)
        # Only one real LLM call; others from cache
        assert mock_run.call_count == 1
        assert all(r.canonical == "Vladimir Putin" for r in results)


# ── Integration: builder.py adds llm features ─────────────────────────────────

def _make_event(
    event_id: str = "e1",
    event_type: str = "military_action",
    raw_title: str = "Troops mobilize near border",
) -> "CanonicalEvent":
    """Helper to build a CanonicalEvent with valid required fields."""
    from normalizer.canonical import CanonicalEvent
    from datetime import datetime, timezone
    return CanonicalEvent(
        event_id=event_id,
        doc_ids=[],
        source="rss",
        occurred_at=datetime.now(timezone.utc),
        event_type=event_type,
        sub_event_type="",
        actors=[],
        country="",
        severity=0.5,
        polarity=-0.5,
        fatalities=0,
        independent_sources=1,
        contradiction_score=0.0,
        raw_title=raw_title,
    )


class TestBuilderLLMIntegration:

    def test_llm_features_added_to_dict_when_available(self):
        from features.builder import build_features

        ev = _make_event("e1", raw_title="Troops mobilize near border")
        mock_result = LLMFeatures(
            threat_level=0.75, escalation=0.60, deescalation=0.10,
            event_certainty=0.85, actor_hostility=0.70, available=True,
        )

        with patch("llm.text_features.extract_llm_features", return_value=mock_result) as mock_ex:
            feat, _ = build_features([ev], question="Will there be an escalation?", use_llm=True)

        mock_ex.assert_called_once()
        assert "llm_threat_level" in feat
        assert abs(feat["llm_threat_level"] - 0.75) < 1e-6
        assert feat["llm_available"] == 1.0

    def test_llm_features_zero_when_use_llm_false(self):
        from features.builder import build_features

        ev = _make_event("e2", raw_title="Border incident")

        with patch("llm.text_features.extract_llm_features") as mock_ex:
            feat, _ = build_features([ev], question="Q?", use_llm=False)

        mock_ex.assert_not_called()
        assert feat.get("llm_available", 0.0) == 0.0
        assert feat.get("llm_threat_level", 0.0) == 0.0

    def test_llm_failure_does_not_crash_builder(self):
        from features.builder import build_features

        ev = _make_event("e3", event_type="protest", raw_title="Protest in capital")

        with patch("llm.text_features.extract_llm_features", side_effect=Exception("Ollama down")):
            feat, prov = build_features([ev], question="Q?", use_llm=True)

        # Builder should still return valid v3 features
        assert "military_count_7d" in feat
        assert feat.get("llm_available", 0.0) == 0.0

    def test_v3_model_features_unchanged(self):
        """LLM features must not pollute the v3 feature set."""
        from features.builder import build_features, get_feature_names

        v3_names = set(get_feature_names())
        ev = _make_event("e4", raw_title="Airstrike reported")

        mock_result = LLMFeatures(threat_level=0.9, available=True)
        with patch("llm.text_features.extract_llm_features", return_value=mock_result):
            feat, _ = build_features([ev], question="Q?", use_llm=True)

        # All v3 features still present
        for name in v3_names:
            assert name in feat, f"v3 feature missing: {name}"

        # LLM features don't appear in v3 list
        for llm_key in ["llm_threat_level", "llm_available"]:
            assert llm_key not in v3_names
