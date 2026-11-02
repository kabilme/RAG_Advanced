"""Hybrid retrieval engine combining semantic and BM25 lexical search using Reciprocal Rank Fusion (RRF)."""

from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.retrieval.base import RetrievedChunk
from app.retrieval.semantic_search import SemanticSearchEngine
from app.retrieval.keyword_search import BM25SearchEngine


class HybridSearchEngine:
    """Fuses dense vector search and sparse BM25 search via Reciprocal Rank Fusion."""

    def __init__(
        self,
        semantic_engine: SemanticSearchEngine,
        keyword_engine: BM25SearchEngine,
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        self.semantic_engine = semantic_engine
        self.keyword_engine = keyword_engine
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Runs parallel/sequential semantic and lexical searches and merges results using RRF."""
        k = top_k or settings.TOP_K
        candidate_pool_size = max(k * 2, 20)

        # 1. Fetch dense candidates
        dense_results = self.semantic_engine.search(
            query=query,
            top_k=candidate_pool_size,
            similarity_threshold=similarity_threshold,
            filter_criteria=filter_criteria,
        )

        # 2. Fetch sparse candidates
        sparse_results = self.keyword_engine.search(
            query=query,
            top_k=candidate_pool_size,
            filter_criteria=filter_criteria,
        )

        # If one engine returns empty, fallback to the other
        if not dense_results and not sparse_results:
            return []
        if not dense_results:
            return sparse_results[:k]
        if not sparse_results:
            return dense_results[:k]

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF_score(d) = sum( weight_m / (k + rank_m(d)) )
        chunk_map: Dict[str, RetrievedChunk] = {}
        rrf_scores: Dict[str, float] = {}

        for rank, chunk in enumerate(dense_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                self.dense_weight / (self.rrf_k + rank)
            )

        for rank, chunk in enumerate(sparse_results, start=1):
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                self.sparse_weight / (self.rrf_k + rank)
            )

        # 4. Sort candidates by RRF score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        top_chunk_ids = sorted_chunk_ids[:k]

        # Theoretical maximum RRF score (rank 1 in both models)
        max_possible_score = (self.dense_weight / (self.rrf_k + 1)) + (self.sparse_weight / (self.rrf_k + 1))

        fused_candidates: List[RetrievedChunk] = []
        for rank, cid in enumerate(top_chunk_ids, start=1):
            base_chunk = chunk_map[cid]
            raw_rrf = rrf_scores[cid]
            norm_score = min(1.0, raw_rrf / max_possible_score if max_possible_score > 0 else raw_rrf)

            fused_candidates.append(
                RetrievedChunk(
                    chunk_id=base_chunk.chunk_id,
                    document_id=base_chunk.document_id,
                    text=base_chunk.text,
                    score=norm_score,
                    rank=rank,
                    retrieval_method="hybrid",
                    filename=base_chunk.filename,
                    page_number=base_chunk.page_number,
                    slide_number=base_chunk.slide_number,
                    sheet_name=base_chunk.sheet_name,
                    section=base_chunk.section,
                    source=base_chunk.source,
                    parent_chunk_id=base_chunk.parent_chunk_id,
                    parent_text=base_chunk.parent_text,
                    metadata=base_chunk.metadata,
                )
            )

        logger.debug(
            "Hybrid RRF search merged %d dense + %d sparse into %d candidates",
            len(dense_results), len(sparse_results), len(fused_candidates)
        )
        return fused_candidates
