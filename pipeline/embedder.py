from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# sentence-transformers is optional — fall back to TF-IDF if not installed
try:
    from sentence_transformers import SentenceTransformer as _ST
    _MODEL: Optional[_ST] = None
    EMBEDDINGS_AVAILABLE = True

    def _get_model() -> _ST:
        global _MODEL
        if _MODEL is None:
            logger.debug("embedder: loading all-MiniLM-L6-v2")
            _MODEL = _ST("all-MiniLM-L6-v2")
        return _MODEL

except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.debug("embedder: sentence-transformers not installed, using TF-IDF fallback")


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of text strings into dense vectors.

    Uses sentence-transformers (all-MiniLM-L6-v2) if available,
    falls back to TF-IDF sparse vectors projected to 64 dims otherwise.

    Returns: array of shape (n_texts, embedding_dim)
    """
    if not texts:
        return np.empty((0, 64))

    if EMBEDDINGS_AVAILABLE:
        model = _get_model()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings  # shape: (n, 384) for MiniLM

    # TF-IDF fallback — produces interpretable but less semantic embeddings
    return _tfidf_embed(texts)


def _tfidf_embed(texts: list[str]) -> np.ndarray:
    """Fallback: TF-IDF + SVD projection to 64 dims."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    n = len(texts)
    n_components = min(64, n - 1) if n > 1 else 1

    try:
        vec = TfidfVectorizer(max_features=2000, stop_words="english")
        X = vec.fit_transform(texts)

        if n_components < X.shape[1] and n > 1:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            return svd.fit_transform(X).astype(np.float32)
        else:
            return X.toarray().astype(np.float32)
    except Exception as e:
        logger.warning("embedder: TF-IDF failed (%s), returning zeros", e)
        return np.zeros((n, 64), dtype=np.float32)
