"""Domain-specific exceptions for Local Advanced RAG."""


class RAGException(Exception):
    """Base exception for all RAG system errors."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DocumentValidationError(RAGException):
    """Raised when an uploaded document fails validation (size, extension, signature, empty)."""
    pass


class UnsupportedFileTypeError(DocumentValidationError):
    """Raised when an uploaded file type is not supported."""
    pass


class DocumentExtractionError(RAGException):
    """Raised when text extraction from a document fails."""
    pass


class ChunkingError(RAGException):
    """Raised when dividing document text into chunks fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when generating embeddings via Ollama or Sentence Transformers fails."""
    pass


class VectorStoreError(RAGException):
    """Raised when ChromaDB operations fail."""
    pass


class RetrievalError(RAGException):
    """Raised when document retrieval fails."""
    pass


class OllamaConnectionError(RAGException):
    """Raised when the application cannot connect to the Ollama server."""
    pass


class ModelNotAvailableError(RAGException):
    """Raised when a configured Ollama or embedding model has not been downloaded."""
    pass


class CitationValidationError(RAGException):
    """Raised when citations in an answer fail validation."""
    pass


class SecurityViolationError(RAGException):
    """Raised when a prompt injection or directory traversal attempt is detected."""
    pass
