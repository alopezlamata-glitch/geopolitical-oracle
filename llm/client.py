"""
Async Ollama HTTP client.

All public methods return None / False on any failure — the caller
must never crash because Ollama is not running or the model is not pulled.

Default endpoints:
  GET  /api/tags               — list available models (health check)
  POST /api/generate           — text generation (with JSON schema format)
  POST /api/embeddings         — dense vector embedding

Configuration (via .env or environment):
  OLLAMA_HOST        http://localhost:11434
  OLLAMA_TEXT_MODEL  (auto-detected from best available, or explicit override)
  OLLAMA_EMBED_MODEL nomic-embed-text

Model auto-detection priority (best reasoning → fastest):
  llama3.3 > qwen2.5 > mistral-nemo > llama3.1 > llama3.2 > mistral > llama2
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "http://localhost:11434"
_GENERATE_TIMEOUT = aiohttp.ClientTimeout(total=120)   # larger models need more time
_EMBED_TIMEOUT    = aiohttp.ClientTimeout(total=20)
_HEALTH_TIMEOUT   = aiohttp.ClientTimeout(total=5)

# Model preference order — best available wins.
# Partial name match: "llama3.3" matches "llama3.3:latest", "llama3.3:70b", etc.
_MODEL_PREFERENCE = [
    "llama3.3",
    "qwen2.5",
    "mistral-nemo",
    "llama3.1",
    "llama3.2",
    "mistral",
    "llama2",
    "phi3",
    "gemma2",
]

# ── Module-level singleton ────────────────────────────────────────────────────
_client: Optional["OllamaClient"] = None


def get_client() -> "OllamaClient":
    """Return the module-level singleton OllamaClient."""
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client


def _best_model(available: list[str], fallback: str) -> str:
    """Return the highest-priority model from the preference list that is available."""
    available_lower = [m.lower() for m in available]
    for pref in _MODEL_PREFERENCE:
        for i, name in enumerate(available_lower):
            if name.startswith(pref) or pref in name:
                return available[i]
    return fallback


class OllamaClient:
    """
    Thin async wrapper around the Ollama REST API.

    Instantiate once; re-use across coroutines (it does not hold a persistent
    connection — aiohttp sessions are created per-call to avoid event-loop
    ownership issues when called from sync contexts via asyncio.run()).
    """

    def __init__(
        self,
        host: Optional[str] = None,
        text_model: Optional[str] = None,
        embed_model: Optional[str] = None,
    ) -> None:
        self.host = (host or os.getenv("OLLAMA_HOST", _DEFAULT_HOST)).rstrip("/")
        # If OLLAMA_TEXT_MODEL is set, use it; otherwise auto-detect at first call.
        self._text_model_override = text_model or os.getenv("OLLAMA_TEXT_MODEL")
        self.embed_model = embed_model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self._resolved_text_model: Optional[str] = self._text_model_override

    @property
    def text_model(self) -> str:
        return self._resolved_text_model or "llama3.2"

    # ── Health ─────────────────────────────────────────────────────────────────

    async def is_available(self) -> bool:
        """Return True if Ollama is reachable and has at least one model pulled."""
        try:
            async with aiohttp.ClientSession(timeout=_HEALTH_TIMEOUT) as sess:
                async with sess.get(f"{self.host}/api/tags") as resp:
                    if resp.status != 200:
                        return False
                    data = await resp.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    logger.debug("ollama: available models: %s", models)
                    if models and not self._text_model_override:
                        self._resolved_text_model = _best_model(models, models[0])
                        logger.info("ollama: auto-selected text model: %s", self._resolved_text_model)
                    return bool(models)
        except Exception as e:
            logger.debug("ollama: not available (%s)", e)
            return False

    async def list_models(self) -> list[str]:
        """Return list of pulled model names, empty list on error."""
        try:
            async with aiohttp.ClientSession(timeout=_HEALTH_TIMEOUT) as sess:
                async with sess.get(f"{self.host}/api/tags") as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
                    return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []

    # ── Text generation ────────────────────────────────────────────────────────

    async def generate_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        system: Optional[str] = None,
        schema: Optional[dict] = None,
        retries: int = 2,
    ) -> Optional[dict]:
        """
        Send a generation request with structured JSON output.

        If `schema` is provided (JSON Schema dict), uses Ollama's structured
        output format (Ollama >= 0.4). Falls back to format="json" for older
        versions.

        Returns parsed JSON dict on success, None on any error.
        temperature=0.0 (greedy) for deterministic feature extraction.
        Retries up to `retries` times on 503 (model loading).
        """
        payload: dict = {
            "model": model or self.text_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        # Use JSON schema if provided (Ollama >= 0.4), else basic json mode
        if schema:
            payload["format"] = schema
        else:
            payload["format"] = "json"

        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=_GENERATE_TIMEOUT) as sess:
                    async with sess.post(
                        f"{self.host}/api/generate",
                        json=payload,
                    ) as resp:
                        if resp.status == 503:
                            # Model is loading — wait and retry
                            wait = 3 * (attempt + 1)
                            logger.info(
                                "ollama: 503 model loading, retry %d/%d in %ds",
                                attempt + 1, retries, wait,
                            )
                            await asyncio.sleep(wait)
                            continue
                        if resp.status != 200:
                            body = await resp.text()
                            logger.warning("ollama generate: HTTP %d -- %s", resp.status, body[:200])
                            return None
                        data = await resp.json()
                        raw_text = data.get("response", "")
                        return json.loads(raw_text)
            except json.JSONDecodeError as e:
                logger.warning("ollama generate: JSON parse error -- %s", e)
                return None
            except Exception as e:
                last_error = e
                if attempt < retries:
                    await asyncio.sleep(2)
                    continue
                logger.warning("ollama generate: error -- %s", e)
                return None

        if last_error:
            logger.warning("ollama generate: all retries failed -- %s", last_error)
        return None

    # ── Embeddings ─────────────────────────────────────────────────────────────

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> Optional[list[float]]:
        """
        Return a dense embedding vector for `text`.
        Returns None on any error.
        """
        payload = {
            "model": model or self.embed_model,
            "prompt": text,
        }
        try:
            async with aiohttp.ClientSession(timeout=_EMBED_TIMEOUT) as sess:
                async with sess.post(
                    f"{self.host}/api/embeddings",
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        logger.warning("ollama embed: HTTP %d", resp.status)
                        return None
                    data = await resp.json()
                    emb = data.get("embedding")
                    if isinstance(emb, list) and len(emb) > 0:
                        return [float(x) for x in emb]
                    return None
        except Exception as e:
            logger.warning("ollama embed: error -- %s", e)
            return None
