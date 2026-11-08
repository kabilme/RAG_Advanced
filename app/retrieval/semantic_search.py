"""Semantic vector retrieval engine."""

from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.base import BaseVectorStore
from app.retrieval.base import RetrievedChunk


class SemanticSearchEngine:
    """Performs dense vector similarity search against indexed document chunks."""

    def __init__(self, vector_store: BaseVectorStore, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Generates query vector and retrieves top-K nearest neighbors from vector store."""
        if not query.strip():
            return []

        k = top_k or settings.TOP_K
        threshold = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD

        # Generate query embedding
        query_vec = self.embedding_service.embed_query(query)

        # Query vector store
        raw_results = self.vector_store.similarity_search(
            query_embedding=query_vec,
            top_k=k,
            similarity_threshold=threshold,
            filter_criteria=filter_criteria,
        )

        retrieved: List[RetrievedChunk] = []
        for rank, res in enumerate(raw_results, start=1):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=res.chunk_id,
                    document_id=res.document_id,
                    text=res.text,
                    score=res.similarity_score,
                    rank=rank,
                    retrieval_method="semantic",
                    filename=res.filename,
                    page_number=res.page_number,
                    slide_number=res.slide_number,
                    sheet_name=res.sheet_name,
                    section=res.section,
                    source=res.source,
                    parent_chunk_id=res.parent_chunk_id,
                    parent_text=res.parent_text,
                    context_prefix=res.context_prefix or res.metadata.get("context_prefix"),
                    raw_text=res.raw_text or res.metadata.get("raw_text"),
                    metadata=res.metadata,
                )
            )

        logger.debug("Semantic search for '%s' returned %d chunks", query[:40], len(retrieved))
        return retrieved
