"""Evaluation metrics for retrieval quality: Precision@K, Recall@K, Hit Rate@K, and MRR."""

from typing import List, Set


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculates Precision@K = |Retrieved@K ∩ Relevant| / K."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculates Recall@K = |Retrieved@K ∩ Relevant| / |Relevant|."""
    if not relevant_ids:
        return 1.0 if not retrieved_ids else 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculates Hit Rate@K = 1 if any relevant item in top K, else 0."""
    top_k = retrieved_ids[:k]
    return 1.0 if any(cid in relevant_ids for cid in top_k) else 0.0


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Calculates Reciprocal Rank = 1 / first_relevant_rank."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0
