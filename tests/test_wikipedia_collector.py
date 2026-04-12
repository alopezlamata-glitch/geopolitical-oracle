"""
Tests for collector/wikipedia.py.

Coverage:
  - _keywords: stop-word filtering, truncation to N tokens
  - WikipediaResult defaults
  - collect_wikipedia: found article, disambiguation skip, empty extract, API failure
  - extract_llm_features: wiki_context wired into prompt (no Ollama call)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collector.wikipedia import (
    WikipediaResult,
    _keywords,
    collect_wikipedia,
)


# ── _keywords ─────────────────────────────────────────────────────────────────

class TestKeywords:

    def test_removes_stop_words(self):
        kw = _keywords("Will Iran attack Israel before June?")
        assert "will" not in kw.split()
        assert "before" not in kw.split()

    def test_keeps_meaningful_tokens(self):
        kw = _keywords("Will Iran attack Israel before June?")
        assert "iran" in kw.split()
        assert "israel" in kw.split()
        assert "attack" in kw.split()

    def test_truncates_to_n(self):
        kw = _keywords("alpha beta gamma delta epsilon zeta eta theta", n=3)
        assert len(kw.split()) == 3

    def test_empty_query_returns_empty(self):
        assert _keywords("") == ""

    def test_all_stop_words_returns_empty(self):
        kw = _keywords("will the be is are in on at to for")
        assert kw == ""


# ── WikipediaResult defaults ──────────────────────────────────────────────────

class TestWikipediaResult:

    def test_default_not_found(self):
        r = WikipediaResult()
        assert r.found is False

    def test_default_empty_content(self):
        r = WikipediaResult()
        assert r.content == ""
        assert r.page_title == ""
        assert r.page_url == ""

    def test_found_result(self):
        r = WikipediaResult(
            content="Iran is a country...",
            page_title="Iran",
            page_url="https://en.wikipedia.org/wiki/Iran",
            found=True,
        )
        assert r.found is True
        assert len(r.content) > 0


# ── collect_wikipedia (async, mocked aiohttp) ─────────────────────────────────

def _make_session(search_titles=None, summary_data=None, search_status=200, summary_status=200):
    """Build a mock aiohttp session."""
    session = MagicMock()

    # Mock search response
    search_resp = AsyncMock()
    search_resp.__aenter__ = AsyncMock(return_value=search_resp)
    search_resp.__aexit__ = AsyncMock(return_value=False)
    search_resp.status = search_status
    search_resp.json = AsyncMock(return_value=[
        "query", search_titles or [], [], []
    ])

    # Mock summary response
    summary_resp = AsyncMock()
    summary_resp.__aenter__ = AsyncMock(return_value=summary_resp)
    summary_resp.__aexit__ = AsyncMock(return_value=False)
    summary_resp.status = summary_status
    summary_resp.json = AsyncMock(return_value=summary_data or {})

    # session.get() context manager
    get_cm = MagicMock()
    get_cm.__aenter__ = AsyncMock(side_effect=[search_resp, summary_resp])
    get_cm.__aexit__ = AsyncMock(return_value=False)
    session.get = MagicMock(return_value=get_cm)

    return session


class TestCollectWikipedia:
    """Sync wrappers around async collect_wikipedia — no pytest-asyncio needed."""

    def _run(self, coro):
        import asyncio
        return asyncio.run(coro)

    def test_returns_not_found_when_no_search_results(self):
        session = _make_session(search_titles=[])
        result = self._run(collect_wikipedia(session, "Will Iran attack Israel?"))
        assert result.found is False

    def test_happy_path_returns_content(self):
        summary_data = {
            "type": "standard",
            "title": "Iran",
            "extract": "Iran is a country in Western Asia with a population of 85 million.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Iran"}},
        }
        session = _make_session(search_titles=["Iran"], summary_data=summary_data)
        result = self._run(collect_wikipedia(session, "Will Iran attack Israel?"))
        assert result.found is True
        assert "Iran" in result.content
        assert result.page_title == "Iran"
        assert "wikipedia.org" in result.page_url

    def test_skips_disambiguation_pages(self):
        summary_data = {"type": "disambiguation", "title": "Iran (disambiguation)", "extract": ""}
        session = _make_session(search_titles=["Iran (disambiguation)"], summary_data=summary_data)
        result = self._run(collect_wikipedia(session, "Will Iran attack Israel?"))
        assert result.found is False

    def test_returns_not_found_on_empty_extract(self):
        summary_data = {"type": "standard", "title": "Iran", "extract": ""}
        session = _make_session(search_titles=["Iran"], summary_data=summary_data)
        result = self._run(collect_wikipedia(session, "Will Iran escalate?"))
        assert result.found is False

    def test_truncates_long_content(self):
        long_extract = "X" * 5_000
        summary_data = {
            "type": "standard",
            "title": "Conflict",
            "extract": long_extract,
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Conflict"}},
        }
        session = _make_session(search_titles=["Conflict"], summary_data=summary_data)
        result = self._run(collect_wikipedia(session, "Will there be a conflict?"))
        assert result.found is True
        assert len(result.content) <= 1_500  # MAX_CONTENT_CHARS

    def test_returns_not_found_on_search_http_error(self):
        session = _make_session(search_titles=["Iran"], search_status=500)
        result = self._run(collect_wikipedia(session, "Will Iran attack?"))
        assert result.found is False

    def test_returns_not_found_on_summary_http_error(self):
        session = _make_session(search_titles=["Iran"], summary_status=404)
        result = self._run(collect_wikipedia(session, "Will Iran attack?"))
        assert result.found is False


# ── wiki_context wired into LLM prompt ────────────────────────────────────────

class TestWikiContextInLLMPrompt:

    def test_prompt_includes_wiki_context_when_provided(self):
        """_build_prompt inserts wiki section when wiki_context is non-empty."""
        from llm.text_features import _build_prompt
        prompt = _build_prompt(
            question="Will Iran attack Israel?",
            headlines=["Troops mobilize near border"],
            wiki_context="Iran is a country in Western Asia...",
        )
        assert "Background context (Wikipedia)" in prompt
        assert "Iran is a country" in prompt

    def test_prompt_excludes_wiki_section_when_empty(self):
        """No wiki section injected when wiki_context is empty string."""
        from llm.text_features import _build_prompt
        prompt = _build_prompt(
            question="Will Iran attack Israel?",
            headlines=["Troops mobilize near border"],
            wiki_context="",
        )
        assert "Background context (Wikipedia)" not in prompt

    def test_extract_llm_features_accepts_wiki_context(self):
        """extract_llm_features passes wiki_context through without error."""
        from llm.text_features import extract_llm_features, LLMFeatures

        with patch("llm.text_features.asyncio.run", return_value=LLMFeatures(
            threat_level=0.7, escalation=0.6, available=True
        )):
            result = extract_llm_features(
                question="Will Iran attack?",
                headlines=["Troops near border"],
                wiki_context="Iran is a country in Western Asia.",
            )
        assert result.available is True
        assert result.threat_level == pytest.approx(0.7)
