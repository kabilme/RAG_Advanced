"""Abstract base classes and data transfer models for vector databases."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from app.ingestion.chunking import TextChunk


@dataclass
class SearchResult:
    """Represents a vector similarity search candidate chunk."""
    chunk_id: str
    document_id: str
    text: str
    similarity_score: float
    filename: str
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    sheet_name: Optional[str] = None
    section: Optional[str] = None
    source: str = ""
    parent_chunk_id: Optional[str] = None
    parent_text: Optional[str] = None
    context_prefix: Optional[str] = None
    raw_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """Abstract interface for local persistent vector stores."""

    @abstractmethod
    def add_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> None:
        """Stores chunks, metadata, and vector embeddings."""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Performs vector similarity search against indexed chunks."""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Deletes all chunks associated with a specific document ID."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Returns the total number of indexed vectors."""
        pass

    @abstractmethod
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieves raw chunk metadata and texts for a document."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Deletes all items from the current collection."""
        pass
