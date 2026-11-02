"""Embedding service orchestrator managing providers, dimension validation, and query caching."""

from collections import OrderedDict
from typing import Dict, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import EmbeddingError
from app.embeddings.base import BaseEmbeddingProvider
from app.embeddings.ollama_embeddings import OllamaEmbeddingProvider
from app.embeddings.local_embeddings import SentenceTransformerEmbeddingProvider


class EmbeddingService:
    """Central orchestrator for document and query embedding generation."""

    def __init__(
        self,
        provider_type: Optional[str] = None,
        model_name: Optional[str] = None,
        cache_size: int = 500,
        fallback_to_local: bool = True,
    ):
        self._provider_type = provider_type or ("ollama" if settings.EMBEDDING_MODEL else "sentence_transformers")
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._cache_size = cache_size
        self._fallback_to_local = fallback_to_local
        self._query_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._stats = {"queries_embedded": 0, "cache_hits": 0, "documents_embedded": 0}

        self._provider = self._initialize_provider()

    def _initialize_provider(self) -> BaseEmbeddingProvider:
        """Selects and initializes the embedding provider with automatic fallback if necessary."""
        if self._provider_type.lower() == "ollama":
            ollama_prov = OllamaEmbeddingProvider(model_name=self._model_name)
            if ollama_prov.check_health():
                logger.info("Using Ollama embedding provider with model: %s", self._model_name)
                return ollama_prov
            elif self._fallback_to_local:
                logger.warning(
                    "Ollama model '%s' unavailable. Gracefully falling back to local SentenceTransformer 'all-MiniLM-L6-v2'.",
                    self._model_name
                )
                return SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
            else:
                return ollama_prov
        else:
            local_model = self._model_name if self._model_name and not self._model_name.endswith(":latest") else "all-MiniLM-L6-v2"
            logger.info("Using SentenceTransformer provider with model: %s", local_model)
            return SentenceTransformerEmbeddingProvider(model_name=local_model)

    @property
    def provider(self) -> BaseEmbeddingProvider:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for document chunks with dimension validation."""
        if not texts:
            return []

        embeddings = self._provider.embed_documents(texts)

        # Validate vector dimensions
        expected_dim = self._provider.dimension
        for i, vec in enumerate(embeddings):
            if len(vec) != expected_dim:
                raise EmbeddingError(
                    f"Vector dimension mismatch at index {i}: expected {expected_dim}, got {len(vec)}"
                )

        self._stats["documents_embedded"] += len(texts)
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embeds search query with LRU caching."""
        normalized_query = query.strip()
        if not normalized_query:
            raise EmbeddingError("Cannot embed an empty query.")

        # Check LRU cache
        if normalized_query in self._query_cache:
            self._stats["cache_hits"] += 1
            self._query_cache.move_to_end(normalized_query)
            return self._query_cache[normalized_query]

        # Generate embedding
        vec = self._provider.embed_query(normalized_query)

        # Dimension check
        expected_dim = self._provider.dimension
        if len(vec) != expected_dim:
            raise EmbeddingError(f"Query vector dimension mismatch: expected {expected_dim}, got {len(vec)}")

        # Store in LRU cache
        self._query_cache[normalized_query] = vec
        if len(self._query_cache) > self._cache_size:
            self._query_cache.popitem(last=False)

        self._stats["queries_embedded"] += 1
        return vec

    def get_stats(self) -> Dict[str, any]:
        """Returns embedding usage statistics."""
        return {
            "provider": self._provider.__class__.__name__,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "queries_embedded": self._stats["queries_embedded"],
            "cache_hits": self._stats["cache_hits"],
            "cache_size": len(self._query_cache),
            "documents_embedded": self._stats["documents_embedded"],
        }
