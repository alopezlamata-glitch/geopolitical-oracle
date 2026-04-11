"""
Semantic embedding module using Ollama.

Generates dense vector embeddings for event headlines and the forecast
question, then computes cosine similarity between them.

The similarity score captures whether the collected evidence is semantically
relevant to what's being asked — a signal that raw keyword counts miss:
  - "Troops mass at border" + "Will Iran invade?" → high similarity
  - "Economic sanctions" + "Will Iran invade?" → lower similarity

Output feature:
  llm_query_event_similarity  — cosine similarity in [0, 1] between the
                                  question embedding and the mean event embedding

Also returns per-event similarity scores for the top-k most relevant events
(stored in the embedding_registry table for future retrieval).

Design:
  - One embed() call per text (Ollama API doesn't batch natively)
  - Up to MAX_EMBED_EVENTS events embedded (most recent first)
  - Falls back to 0.5 (uninformative) when Ollama unavailable
  - Non-fatal: all failures return EmbedResult(available=False)

Typical latency: ~50-200ms per embed call with nomic-embed-text on CPU.
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from llm.client import OllamaClient, get_client

logger = logging.getLogger(__name__)

MAX_EMBED_EVENTS = 20   # max events to embed per prediction call


@dataclass
class EmbedResult:
    """
    Output of embed_events_and_question().

    query_similarity: cosine similarity between question and mean event embedding.
    top_event_similarities: list of (event_title, similarity) for the most relevant events.
    available: False when Ollama was not reachable.
    """
    query_similarity: float = 0.5        # 0-1 (0.5 = uninformative neutral)
    top_event_similarities: list[tuple[str, float]] = field(default_factory=list)
    question_embedding: Optional[list[float]] = None
    mean_event_embedding: Optional[list[float]] = None
    n_embedded: int = 0
    latency_ms: Optional[float] = None
    model_used: Optional[str] = None
    available: bool = False


# ── Math helpers ───────────────────────────────────────────────────────────────

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two vectors, clipped to [0, 1].

    Returns 0.5 for zero vectors (uninformative, not undefined).
    """
    if not a or not b or len(a) != len(b):
        return 0.5
    na, nb = _norm(a), _norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.5
    raw = _dot(a, b) / (na * nb)
    # Clip to [0, 1] — angle similarity, not signed similarity
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


def _mean_vector(vecs: list[list[float]]) -> list[float]:
    """Element-wise mean of a list of equal-length vectors."""
    if not vecs:
        return []
    dim = len(vecs[0])
    result = [0.0] * dim
    for v in vecs:
        for i, x in enumerate(v):
            result[i] += x
    n = len(vecs)
    return [x / n for x in result]


# ── Async core ─────────────────────────────────────────────────────────────────

async def _embed_texts_async(
    texts: list[str],
    client: OllamaClient,
) -> list[Optional[list[float]]]:
    """Embed a list of texts sequentially. Returns None for any failed embed."""
    results: list[Optional[list[float]]] = []
    for text in texts:
        vec = await client.embed(text)
        results.append(vec)
    return results


async def _embed_all_async(
    question: str,
    headlines: list[str],
    client: OllamaClient,
) -> EmbedResult:
    import time
    t0 = time.monotonic()

    # Embed question
    q_vec = await client.embed(question[:500])
    if q_vec is None:
        return EmbedResult(available=False)

    # Embed events (most recent first, up to MAX_EMBED_EVENTS)
    selected = headlines[:MAX_EMBED_EVENTS]
    event_vecs_raw = await _embed_texts_async(selected, client)

    # Filter out failed embeddings
    event_vecs = [v for v in event_vecs_raw if v is not None]
    if not event_vecs:
        # Question embedded OK but no event embeddings — use neutral similarity
        latency_ms = (time.monotonic() - t0) * 1000
        return EmbedResult(
            query_similarity=0.5,
            question_embedding=q_vec,
            n_embedded=0,
            latency_ms=round(latency_ms, 1),
            model_used=client.embed_model,
            available=True,
        )

    mean_vec = _mean_vector(event_vecs)
    query_sim = cosine_similarity(q_vec, mean_vec)

    # Per-event similarities (for top-k display)
    per_event = [
        (selected[i], cosine_similarity(q_vec, v))
        for i, v in enumerate(event_vecs_raw)
        if v is not None
    ]
    per_event.sort(key=lambda x: x[1], reverse=True)

    latency_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "embedder: query_sim=%.3f  n_embedded=%d  latency=%.0fms  model=%s",
        query_sim, len(event_vecs), latency_ms, client.embed_model,
    )
    return EmbedResult(
        query_similarity=round(query_sim, 4),
        top_event_similarities=per_event[:5],
        question_embedding=q_vec,
        mean_event_embedding=mean_vec,
        n_embedded=len(event_vecs),
        latency_ms=round(latency_ms, 1),
        model_used=client.embed_model,
        available=True,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def embed_events_and_question(
    question: str,
    headlines: list[str],
    client: Optional[OllamaClient] = None,
) -> EmbedResult:
    """
    Compute semantic similarity between the forecast question and event headlines.

    Synchronous wrapper — safe to call from non-async code (features/builder.py).
    Returns EmbedResult(available=False, query_similarity=0.5) on any failure.

    Args:
        question  : the binary question being forecast
        headlines : event titles (most recent first)
        client    : OllamaClient; uses module singleton if None
    """
    if not question or not headlines:
        return EmbedResult(available=False)

    if client is None:
        client = get_client()

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            logger.warning("embedder: called from running event loop — skipping")
            return EmbedResult(available=False)

        return asyncio.run(_embed_all_async(question, headlines, client))
    except Exception as e:
        logger.warning("embedder: failed: %s", e)
        return EmbedResult(available=False)
