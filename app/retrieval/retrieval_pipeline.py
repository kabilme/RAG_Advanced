"""Full multi-stage retrieval pipeline orchestrator."""

from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.retrieval.base import RetrievedChunk
from app.retrieval.semantic_search import SemanticSearchEngine
from app.retrieval.keyword_search import BM25SearchEngine
from app.retrieval.hybrid_search import HybridSearchEngine
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.context_compressor import ContextCompressor


class RetrievalPipeline:
    """Orchestrates end-to-end document retrieval from query understanding to context compression."""

    def __init__(
        self,
        hybrid_engine: HybridSearchEngine,
        reranker: Optional[CrossEncoderReranker] = None,
        strategy: Optional[str] = None,
    ):
        self.hybrid_engine = hybrid_engine
        self.reranker = reranker or CrossEncoderReranker()
        self.strategy = strategy or settings.RETRIEVAL_STRATEGY

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[RetrievedChunk]:
        """Executes the complete multi-stage retrieval pipeline.
        
        Stages:
            1. Query understanding and follow-up coreference resolution.
            2. Multi-query decomposition if question is compound.
            3. Retrieval execution (hybrid, semantic, or keyword).
            4. Result merging and RRF deduplication.
            5. Cross-encoder reranking (if enabled).
            6. Context compression and redundant chunk filtering.
        """
        if not query.strip():
            return []

        k = top_k or settings.TOP_K

        # Stage 1: Resolve coreferences and follow-up questions
        resolved_query = QueryRewriter.resolve_followup(query, conversation_history)
        query_type = QueryRewriter.classify_query(resolved_query)
        logger.info("Query '%s' classified as '%s' (Resolved: '%s')", query, query_type, resolved_query)

        # Stage 2: Subquery decomposition for multi-part questions
        subqueries = QueryRewriter.decompose_query(resolved_query)

        # Stage 3: Retrieval execution
        all_candidates: List[RetrievedChunk] = []
        candidate_ids = set()

        candidate_pool_size = max(k * 4, 30)

        for sq in subqueries:
            if self.strategy == "semantic":
                candidates = self.hybrid_engine.semantic_engine.search(
                    query=sq,
                    top_k=candidate_pool_size,
                    similarity_threshold=similarity_threshold,
                    filter_criteria=filter_criteria,
                )
            elif self.strategy == "keyword":
                candidates = self.hybrid_engine.keyword_engine.search(
                    query=sq,
                    top_k=candidate_pool_size,
                    filter_criteria=filter_criteria,
                )
            else:  # Hybrid default
                candidates = self.hybrid_engine.search(
                    query=sq,
                    top_k=candidate_pool_size,
                    similarity_threshold=similarity_threshold,
                    filter_criteria=filter_criteria,
                )

            for c in candidates:
                if c.chunk_id not in candidate_ids:
                    all_candidates.append(c)
                    candidate_ids.add(c.chunk_id)

        if not all_candidates:
            logger.info("No candidate chunks retrieved for query '%s'", resolved_query)
            return []

        # Stage 4: Cross-Encoder Reranking
        if settings.ENABLE_RERANKER and self.reranker:
            rerank_k = top_k or settings.RERANK_TOP_K or k
            reranked = self.reranker.rerank(
                query=resolved_query,
                candidates=all_candidates,
                top_k=rerank_k,
            )
        else:
            reranked = all_candidates[:k]

        # Stage 5: Context compression and near-duplicate removal
        compressed = ContextCompressor.compress(resolved_query, reranked)

        logger.info("Retrieval pipeline completed: %d final chunks returned", len(compressed))
        return compressed
