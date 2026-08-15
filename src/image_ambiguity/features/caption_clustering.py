"""Semantic caption clustering via Sentence-BERT cosine similarity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class CaptionClusterResult:
    """Clustered captions with metadata for debugging / API payloads."""

    representatives: list[str]
    cluster_ids: list[int]
    n_input: int
    n_clusters: int
    similarity_threshold: float
    applied: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


def cluster_captions_by_similarity(
    captions: Sequence[str],
    embeddings: np.ndarray,
    *,
    similarity_threshold: float = 0.85,
    apply: bool = True,
) -> CaptionClusterResult:
    """Greedy semantic clustering; return one representative per cluster.

    Captions with pairwise cosine similarity ``>= similarity_threshold`` are
    merged. The representative is the caption closest to the cluster mean
    embedding (first member as tie-break).

    If clustering yields a single representative but multiple inputs exist,
    the two inputs farthest apart *within that cluster* are kept so diversity
    can still be computed without inventing text. If only one caption exists,
    it is duplicated so downstream diversity code always has ``n >= 2``.

    Args:
        captions: Caption strings (length ``n``).
        embeddings: Array of shape ``(n, dim)``.
        similarity_threshold: Merge if cosine similarity is at least this.
        apply: When false, return the cleaned input list unchanged (still
            ensuring at least two strings for diversity).

    Returns:
        :class:`CaptionClusterResult` with representative captions.
    """
    cleaned = [" ".join(str(c).strip().split()) for c in captions if str(c).strip()]
    if not cleaned:
        raise ValueError("At least one non-empty caption is required")

    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != len(cleaned):
        raise ValueError(
            f"embeddings shape {array.shape} does not match "
            f"{len(cleaned)} captions"
        )

    if not apply or len(cleaned) == 1:
        reps = list(cleaned)
        if len(reps) == 1:
            reps = [reps[0], reps[0]]
        return CaptionClusterResult(
            representatives=reps,
            cluster_ids=list(range(len(cleaned))),
            n_input=len(cleaned),
            n_clusters=len(cleaned),
            similarity_threshold=float(similarity_threshold),
            applied=False,
        )

    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1]")

    normalized = _l2_normalize(array)
    similarity = normalized @ normalized.T
    n = len(cleaned)
    assigned = [-1] * n
    clusters: list[list[int]] = []

    for index in range(n):
        if assigned[index] >= 0:
            continue
        cluster_id = len(clusters)
        members = [index]
        assigned[index] = cluster_id
        for other in range(index + 1, n):
            if assigned[other] >= 0:
                continue
            if float(similarity[index, other]) >= similarity_threshold:
                assigned[other] = cluster_id
                members.append(other)
        clusters.append(members)

    representatives: list[str] = []
    for members in clusters:
        member_emb = normalized[members]
        centroid = member_emb.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
        scores = member_emb @ centroid
        best_local = int(np.argmax(scores))
        representatives.append(cleaned[members[best_local]])

    # Ensure >= 2 captions for diversity metrics.
    if len(representatives) == 1:
        members = clusters[0]
        if len(members) >= 2:
            # Farthest pair inside the only cluster (still high similarity).
            sub = similarity[np.ix_(members, members)]
            # Minimize similarity among distinct pairs.
            tril = np.triu_indices(len(members), k=1)
            pair_idx = int(np.argmin(sub[tril]))
            i, j = tril[0][pair_idx], tril[1][pair_idx]
            representatives = [cleaned[members[i]], cleaned[members[j]]]
        else:
            representatives = [representatives[0], representatives[0]]

    return CaptionClusterResult(
        representatives=representatives,
        cluster_ids=assigned,
        n_input=n,
        n_clusters=len(clusters),
        similarity_threshold=float(similarity_threshold),
        applied=True,
    )
