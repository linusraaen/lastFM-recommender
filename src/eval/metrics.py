"""Ranking metrics for top-k artist retrieval against held-out positives."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def ndcg_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, item in enumerate(recommended[:k], start=1) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_recommended: Iterable[Sequence[str]], catalog_size: int) -> float:
    if catalog_size == 0:
        return 0.0
    recommended_union: set[str] = set()
    for recs in all_recommended:
        recommended_union.update(recs)
    return len(recommended_union) / catalog_size


def evaluate_recommendations(
    recommendations: dict[str, Sequence[str]],
    ground_truth: dict[str, set[str]],
    k: int,
    catalog_size: int,
) -> dict[str, float]:
    """recommendations/ground_truth are keyed by user_id; users absent from
    ground_truth (no held-out positives) are skipped."""
    users = [u for u in ground_truth if ground_truth[u]]
    if not users:
        return {"recall@k": 0.0, "map@k": 0.0, "ndcg@k": 0.0, "coverage": 0.0, "n_users": 0}

    recalls, maps, ndcgs = [], [], []
    for u in users:
        recs = recommendations.get(u, [])
        relevant = ground_truth[u]
        recalls.append(recall_at_k(recs, relevant, k))
        maps.append(average_precision_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))

    return {
        "recall@k": sum(recalls) / len(users),
        "map@k": sum(maps) / len(users),
        "ndcg@k": sum(ndcgs) / len(users),
        "coverage": coverage([recommendations.get(u, []) for u in users], catalog_size),
        "n_users": len(users),
    }
