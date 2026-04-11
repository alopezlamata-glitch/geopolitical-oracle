"""
Async Ollama HTTP client.

All public methods return None / False on any failure — the caller
must never crash because Ollama is not running or the model is not pulled.

Default endpoints:
  GET  /api/tags               — list available models (health check)
  POST /api/generate           — text generation (with format:"json")
  POST /api/embeddings         — dense vector embedding

Configuration (via .env or environment):
  OLLAMA_HOST        http://localhost:11434
  OLLAMA_TEXT_MODEL  llama3.2
  OLLAMA_EMBED_MODEL nomic-embed-text
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "http://localhost:11434"
_GENERATE_TIMEOUT = aiohttp.ClientTimeout(total=45)
_EMBED_TIMEOUT    = aiohttp.ClientTimeout(total=20)
_HEALTH_TIMEOUT   = aiohttp.ClientTimeout(total=5)

# ── Module-level singleton ────────────────────────────────────────────────────
_client: Optional["OllamaClient"] = None


def get_client() -> "OllamaClient":
    """Return the module-level singleton OllamaClient."""
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client


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
        self.text_model  = text_model  or os.getenv("OLLAMA_TEXT_MODEL",  "llama3.2")
        self.embed_model = embed_model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

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
                    return True
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
    ) -> Optional[dict]:
        """
        Send a generation request with format="json".

        Returns parsed JSON dict on success, None on any error.
        temperature=0.0 (greedy) for deterministic feature extraction.
        """
        payload: dict = {
            "model": model or self.text_model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        try:
            async with aiohttp.ClientSession(timeout=_GENERATE_TIMEOUT) as sess:
                async with sess.post(
                    f"{self.host}/api/generate",
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning("ollama generate: HTTP %d — %s", resp.status, body[:200])
                        return None
                    data = await resp.json()
                    raw_text = data.get("response", "")
                    return json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.warning("ollama generate: JSON parse error — %s", e)
            return None
        except Exception as e:
            logger.warning("ollama generate: error — %s", e)
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
            logger.warning("ollama embed: error — %s", e)
            return None
