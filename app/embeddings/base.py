"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Abstract interface for local embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or path of the embedding model."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension produced by this model."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of documents/chunks."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generates an embedding vector for a search query."""
        pass
