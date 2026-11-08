"""Unified document ingestion, validation, extraction, and chunking manager."""

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Type
import uuid
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import (
    DocumentValidationError,
    UnsupportedFileTypeError,
    DocumentExtractionError,
    SecurityViolationError,
)
from app.ingestion.loaders import (
    BaseLoader,
    PDFLoader,
    DocxLoader,
    PptxLoader,
    ExcelLoader,
    CSVLoader,
    TextLoader,
    MarkdownLoader,
    HTMLLoader,
    DocumentSection,
)
from app.ingestion.chunking import DocumentChunker, TextChunk


@dataclass
class IngestionResult:
    """The result of ingesting, parsing, and chunking a document."""
    document_id: str
    filename: str
    file_type: str
    document_hash: str
    file_size_bytes: int
    sections: List[DocumentSection]
    chunks: List[TextChunk]
    metadata: Dict[str, any] = field(default_factory=dict)
    is_duplicate: bool = False


class DocumentIngestionEngine:
    """Manages the full lifecycle of document validation, extraction, and chunking."""

    # Map supported extensions to their specialized loader classes
    _LOADER_REGISTRY: Dict[str, Type[BaseLoader]] = {
        ".pdf": PDFLoader,
        ".docx": DocxLoader,
        ".pptx": PptxLoader,
        ".xlsx": ExcelLoader,
        ".xls": ExcelLoader,
        ".csv": CSVLoader,
        ".txt": TextLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        ".html": HTMLLoader,
        ".htm": HTMLLoader,
    }

    # Standard magic byte signatures for security validation
    _FILE_SIGNATURES: Dict[str, bytes] = {
        ".pdf": b"%PDF-",
        ".docx": b"PK\x03\x04",
        ".pptx": b"PK\x03\x04",
        ".xlsx": b"PK\x03\x04",
    }

    def __init__(
        self,
        chunker: Optional[DocumentChunker] = None,
        known_hashes: Optional[Set[str]] = None,
    ):
        self.chunker = chunker or DocumentChunker(
            strategy=settings.CHUNK_STRATEGY,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            enable_parent_child=settings.ENABLE_PARENT_CHILD,
            parent_chunk_size=settings.PARENT_CHUNK_SIZE,
            enable_contextual=getattr(settings, "ENABLE_CONTEXTUAL_RETRIEVAL", True),
            contextual_mode=getattr(settings, "CONTEXTUAL_MODE", "structural"),
        )
        self.known_hashes = known_hashes if known_hashes is not None else set()

    @classmethod
    def calculate_sha256(cls, file_path: Path) -> str:
        """Computes SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    @classmethod
    def validate_file(cls, file_path: Path, max_size_mb: Optional[int] = None) -> str:
        """Validates file safety, extension, size, and header signature.
        
        Returns:
            The normalized file extension (e.g., '.pdf').
            
        Raises:
            SecurityViolationError or DocumentValidationError.
        """
        if not file_path.exists():
            raise DocumentValidationError(f"File does not exist: {file_path}")

        # Path traversal and safe filename check
        clean_name = os.path.basename(file_path.name)
        if ".." in clean_name or "/" in clean_name or "\\" in clean_name:
            raise SecurityViolationError(f"Potential path traversal in filename: {clean_name}")

        ext = file_path.suffix.lower()
        if ext not in cls._LOADER_REGISTRY:
            raise UnsupportedFileTypeError(
                f"Unsupported file format '{ext}'. Supported formats: {list(cls._LOADER_REGISTRY.keys())}"
            )

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise DocumentValidationError(f"Uploaded file '{clean_name}' is empty (0 bytes).")

        max_mb = max_size_mb or settings.MAX_UPLOAD_SIZE_MB
        if file_size > max_mb * 1024 * 1024:
            raise DocumentValidationError(
                f"File '{clean_name}' ({file_size / (1024*1024):.1f} MB) exceeds maximum allowed size of {max_mb} MB."
            )

        # Signature verification for binary formats
        if ext in cls._FILE_SIGNATURES:
            sig = cls._FILE_SIGNATURES[ext]
            with open(file_path, "rb") as f:
                header = f.read(len(sig))
                if header != sig:
                    raise DocumentValidationError(
                        f"File '{clean_name}' signature mismatch for format {ext}."
                    )

        return ext

    def ingest_file(
        self,
        file_path: Path,
        custom_document_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, any]] = None,
    ) -> IngestionResult:
        """Processes a single file: validate, hash, extract, and chunk.
        
        Args:
            file_path: Path to the target document.
            custom_document_id: Optional document UUID.
            custom_metadata: Optional additional metadata dict.
            
        Returns:
            IngestionResult containing parsed sections, chunks, and metadata.
        """
        ext = self.validate_file(file_path)
        file_hash = self.calculate_sha256(file_path)
        clean_filename = file_path.name
        file_size = file_path.stat().st_size
        doc_id = custom_document_id or str(uuid.uuid4())

        # Check for duplicates
        is_duplicate = file_hash in self.known_hashes
        if is_duplicate:
            logger.info("File '%s' with hash %s is already indexed.", clean_filename, file_hash)

        # Instantiate appropriate loader
        loader_cls = self._LOADER_REGISTRY[ext]
        loader = loader_cls()

        # Extract content
        logger.info("Extracting document content for '%s' using %s", clean_filename, loader_cls.__name__)
        sections = loader.load(file_path)

        if not sections or not any(s.text.strip() for s in sections):
            raise DocumentExtractionError(f"No readable text content extracted from '{clean_filename}'.")

        # Chunk content
        chunks = self.chunker.chunk_document(
            document_id=doc_id,
            filename=clean_filename,
            file_type=ext.lstrip("."),
            document_hash=file_hash,
            sections=sections,
        )

        metadata = custom_metadata or {}
        metadata.update({
            "section_count": len(sections),
            "chunk_count": len(chunks),
            "file_size_bytes": file_size,
        })

        self.known_hashes.add(file_hash)

        return IngestionResult(
            document_id=doc_id,
            filename=clean_filename,
            file_type=ext.lstrip("."),
            document_hash=file_hash,
            file_size_bytes=file_size,
            sections=sections,
            chunks=chunks,
            metadata=metadata,
            is_duplicate=is_duplicate,
        )
