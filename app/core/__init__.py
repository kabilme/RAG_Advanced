"""Core application configuration, logging, and exceptions."""

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import (
    RAGException,
    DocumentValidationError,
    UnsupportedFileTypeError,
    DocumentExtractionError,
    ChunkingError,
    EmbeddingError,
    VectorStoreError,
    RetrievalError,
    OllamaConnectionError,
    ModelNotAvailableError,
    CitationValidationError,
    SecurityViolationError,
)

__all__ = [
    "settings",
    "logger",
    "RAGException",
    "DocumentValidationError",
    "UnsupportedFileTypeError",
    "DocumentExtractionError",
    "ChunkingError",
    "EmbeddingError",
    "VectorStoreError",
    "RetrievalError",
    "OllamaConnectionError",
    "ModelNotAvailableError",
    "CitationValidationError",
    "SecurityViolationError",
]
