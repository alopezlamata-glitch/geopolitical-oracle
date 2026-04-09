from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MIN_CLUSTER_SIZE = 2
_EPS_QUANTILE = 0.3  # DBSCAN eps = 30th percentile of pairwise distances


@dataclass
class ClusterResult:
    n_clusters: int
    n_noise: int
    labels: list[int]                      # -1 = noise
    themes: list[str]                      # one summary per cluster
    anomalous_texts: list[str]             # texts not fitting any cluster
    cluster_sizes: list[int]

    def is_empty(self) -> bool:
        return self.n_clusters == 0

    def to_prompt_text(self) -> str:
        if self.is_empty():
            return "(no distinct event clusters detected)"
        lines = []
        for i, (theme, size) in enumerate(zip(self.themes, self.cluster_sizes), 1):
            lines.append(f"Cluster {i} ({size} articles): {theme}")
        if self.anomalous_texts:
            lines.append(f"Isolated signals ({len(self.anomalous_texts)} articles): not fitting main themes")
        return "\n".join(lines)


def cluster_events(
    texts: list[str],
    embeddings: np.ndarray,
) -> ClusterResult:
    """
    Cluster event texts using DBSCAN on their embeddings.
    Returns ClusterResult with themes derived from most representative texts per cluster.
    """
    if embeddings.shape[0] < _MIN_CLUSTER_SIZE:
        return ClusterResult(
            n_clusters=0, n_noise=len(texts), labels=[-1] * len(texts),
            themes=[], anomalous_texts=texts[:5], cluster_sizes=[],
        )

    try:
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import pairwise_distances

        # Compute eps from data distribution
        dists = pairwise_distances(embeddings, metric="cosine")
        np.fill_diagonal(dists, np.inf)
        min_dists = dists.min(axis=1)
        eps = float(np.quantile(min_dists[np.isfinite(min_dists)], _EPS_QUANTILE))
        eps = max(eps, 0.05)  # floor to prevent degenerate eps

        db = DBSCAN(eps=eps, min_samples=_MIN_CLUSTER_SIZE, metric="cosine")
        labels = db.fit_predict(embeddings)

        unique_labels = sorted(set(labels) - {-1})
        n_clusters = len(unique_labels)
        n_noise = int((labels == -1).sum())

        themes = []
        cluster_sizes = []
        for label in unique_labels:
            mask = labels == label
            cluster_texts = [t for t, m in zip(texts, mask) if m]
            cluster_embs = embeddings[mask]

            # Representative text = closest to cluster centroid
            centroid = cluster_embs.mean(axis=0, keepdims=True)
            dists_to_centroid = pairwise_distances(cluster_embs, centroid, metric="cosine").flatten()
            rep_idx = int(dists_to_centroid.argmin())
            rep_text = cluster_texts[rep_idx]

            # Theme = first 100 chars of most representative text
            themes.append(rep_text[:100].strip())
            cluster_sizes.append(int(mask.sum()))

        anomalous = [t for t, lab in zip(texts, labels) if lab == -1][:5]

        return ClusterResult(
            n_clusters=n_clusters,
            n_noise=n_noise,
            labels=labels.tolist(),
            themes=themes,
            anomalous_texts=anomalous,
            cluster_sizes=cluster_sizes,
        )

    except Exception as e:
        logger.warning("clusterer: failed (%s)", e)
        return ClusterResult(
            n_clusters=0, n_noise=len(texts), labels=[-1] * len(texts),
            themes=[], anomalous_texts=texts[:3], cluster_sizes=[],
        )
