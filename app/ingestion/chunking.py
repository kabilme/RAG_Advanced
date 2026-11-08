"""Intelligent document chunking engine supporting multiple chunking strategies."""

import re
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import ChunkingError
from app.ingestion.loaders import DocumentSection


@dataclass
class TextChunk:
    """Represents a discrete text chunk with traceable metadata."""
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    filename: str
    file_type: str
    document_hash: str
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

    def to_dict(self) -> Dict[str, Any]:
        """Converts chunk metadata and content to a dictionary for vector storage."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "page_number": self.page_number if self.page_number is not None else -1,
            "slide_number": self.slide_number if self.slide_number is not None else -1,
            "sheet_name": self.sheet_name or "",
            "section": self.section or "",
            "chunk_index": self.chunk_index,
            "document_hash": self.document_hash,
            "source": self.source,
            "parent_chunk_id": self.parent_chunk_id or "",
            "context_prefix": self.context_prefix or "",
            "raw_text": self.raw_text or "",
        }


class DocumentChunker:
    """Splits document sections into intelligent, traceable chunks."""

    def __init__(
        self,
        strategy: str = "structure_aware",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        enable_parent_child: bool = True,
        parent_chunk_size: int = 2500,
        enable_contextual: Optional[bool] = None,
        contextual_mode: Optional[str] = None,
        ollama_client: Optional[Any] = None,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_parent_child = enable_parent_child
        self.parent_chunk_size = parent_chunk_size
        self.enable_contextual = (
            enable_contextual if enable_contextual is not None else getattr(settings, "ENABLE_CONTEXTUAL_RETRIEVAL", True)
        )
        self.contextual_mode = contextual_mode or getattr(settings, "CONTEXTUAL_MODE", "structural")
        self.ollama_client = ollama_client

    def chunk_document(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        document_hash: str,
        sections: List[DocumentSection],
    ) -> List[TextChunk]:
        """Divides document sections into chunks according to configured strategy."""
        if not sections:
            return []

        chunks: List[TextChunk] = []
        global_chunk_index = 0

        if self.strategy == "structure_aware":
            chunks = self._chunk_structure_aware(
                document_id, filename, file_type, document_hash, sections
            )
        elif self.strategy == "recursive_character":
            chunks = self._chunk_recursive(
                document_id, filename, file_type, document_hash, sections
            )
        elif self.strategy == "sentence":
            chunks = self._chunk_sentence(
                document_id, filename, file_type, document_hash, sections
            )
        elif self.strategy == "paragraph":
            chunks = self._chunk_paragraph(
                document_id, filename, file_type, document_hash, sections
            )
        else:
            chunks = self._chunk_fixed(
                document_id, filename, file_type, document_hash, sections
            )

        # Attach parent-child context if enabled
        if self.enable_parent_child:
            chunks = self._enrich_parent_child_context(chunks)

        # Attach Contextual Retrieval situational preambles if enabled
        if self.enable_contextual:
            chunks = self._enrich_contextual_retrieval(chunks, filename, sections)

        logger.debug(
            "Created %d chunks for document %s using strategy '%s' (Contextual: %s)",
            len(chunks), filename, self.strategy, self.enable_contextual
        )
        return chunks

    def _enrich_contextual_retrieval(
        self,
        chunks: List[TextChunk],
        filename: str,
        sections: List[DocumentSection],
    ) -> List[TextChunk]:
        """Applies Contextual Retrieval prepending document & section situating preambles."""
        from app.ingestion.contextual_enricher import ContextualEnricher
        return ContextualEnricher.enrich_chunks(
            chunks=chunks,
            filename=filename,
            sections=sections,
            mode=self.contextual_mode,
            ollama_client=self.ollama_client,
        )

    def _chunk_structure_aware(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        document_hash: str,
        sections: List[DocumentSection],
    ) -> List[TextChunk]:
        """Preserves section headings, slides, sheets, and pages while splitting oversized text."""
        chunks: List[TextChunk] = []
        chunk_index = 0

        for section in sections:
            text = section.text.strip()
            if not text:
                continue

            # If section text fits comfortably within chunk_size, keep intact
            if len(text) <= self.chunk_size:
                chunks.append(
                    TextChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        text=text,
                        chunk_index=chunk_index,
                        filename=filename,
                        file_type=file_type,
                        document_hash=document_hash,
                        page_number=section.page_number,
                        slide_number=section.slide_number,
                        sheet_name=section.sheet_name,
                        section=section.section_title,
                        source=filename,
                    )
                )
                chunk_index += 1
            else:
                # Oversized section: split by recursive boundaries
                sub_texts = self._recursive_split_text(text, self.chunk_size, self.chunk_overlap)
                for sub in sub_texts:
                    if sub.strip():
                        chunks.append(
                            TextChunk(
                                chunk_id=str(uuid.uuid4()),
                                document_id=document_id,
                                text=sub.strip(),
                                chunk_index=chunk_index,
                                filename=filename,
                                file_type=file_type,
                                document_hash=document_hash,
                                page_number=section.page_number,
                                slide_number=section.slide_number,
                                sheet_name=section.sheet_name,
                                section=section.section_title,
                                source=filename,
                            )
                        )
                        chunk_index += 1

        return chunks

    def _chunk_recursive(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        document_hash: str,
        sections: List[DocumentSection],
    ) -> List[TextChunk]:
        """Splits text using hierarchical separators: double newline, newline, period, space."""
        chunks: List[TextChunk] = []
        chunk_index = 0

        for section in sections:
            sub_texts = self._recursive_split_text(section.text, self.chunk_size, self.chunk_overlap)
            for sub in sub_texts:
                if sub.strip():
                    chunks.append(
                        TextChunk(
                            chunk_id=str(uuid.uuid4()),
                            document_id=document_id,
                            text=sub.strip(),
                            chunk_index=chunk_index,
                            filename=filename,
                            file_type=file_type,
                            document_hash=document_hash,
                            page_number=section.page_number,
                            slide_number=section.slide_number,
                            sheet_name=section.sheet_name,
                            section=section.section_title,
                            source=filename,
                        )
                    )
                    chunk_index += 1
        return chunks

    def _chunk_sentence(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        document_hash: str,
        sections: List[DocumentSection],
    ) -> List[TextChunk]:
        """Splits text by sentence boundaries and bundles until chunk_size is reached."""
        chunks: List[TextChunk] = []
        chunk_index = 0
        sentence_regex = re.compile(r"(?<=[.!?])\s+")

        for section in sections:
            sentences = sentence_regex.split(section.text)
            current_chunk: List[str] = []
            current_len = 0

            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if current_len + len(sent) > self.chunk_size and current_chunk:
                    chunk_str = " ".join(current_chunk)
                    chunks.append(
                        TextChunk(
                            chunk_id=str(uuid.uuid4()),
                            document_id=document_id,
                            text=chunk_str,
                            chunk_index=chunk_index,
                            filename=filename,
                            file_type=file_type,
                            document_hash=document_hash,
                            page_number=section.page_number,
                            slide_number=section.slide_number,
                            sheet_name=section.sheet_name,
                            section=section.section_title,
                            source=filename,
                        )
                    )
                    chunk_index += 1
                    # Overlap handling: retain last sentence if feasible
                    if self.chunk_overlap > 0 and len(current_chunk) > 1:
                        current_chunk = [current_chunk[-1], sent]
                        current_len = len(current_chunk[0]) + len(sent)
                    else:
                        current_chunk = [sent]
                        current_len = len(sent)
                else:
                    current_chunk.append(sent)
                    current_len += len(sent)

            if current_chunk:
                chunks.append(
                    TextChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        text=" ".join(current_chunk),
                        chunk_index=chunk_index,
                        filename=filename,
                        file_type=file_type,
                        document_hash=document_hash,
                        page_number=section.page_number,
                        slide_number=section.slide_number,
                        sheet_name=section.sheet_name,
                        section=section.section_title,
                        source=filename,
                    )
                )
                chunk_index += 1

        return chunks

    def _chunk_paragraph(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        document_hash: str,
        sections: List[DocumentSection],
    ) -> List[TextChunk]:
        """Splits text by double newlines (paragraphs)."""
        chunks: List[TextChunk] = []
        chunk_index = 0

        for section in sections:
            paragraphs = section.text.split("\n\n")
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                chunks.append(
                    TextChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        text=para,
                        chunk_index=chunk_index,
                        filename=filename,
                        file_type=file_type,
                        document_hash=document_hash,
                        page_number=section.page_number,
                        slide_number=section.slide_number,
                        sheet_name=section.sheet_name,
                        section=section.section_title,
                        source=filename,
                    )
                )
                chunk_index += 1

        return chunks

    def _chunk_fixed(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        document_hash: str,
        sections: List[DocumentSection],
    ) -> List[TextChunk]:
        """Fixed-size character windowing."""
        chunks: List[TextChunk] = []
        chunk_index = 0
        step = max(1, self.chunk_size - self.chunk_overlap)

        for section in sections:
            text = section.text
            for i in range(0, len(text), step):
                window = text[i : i + self.chunk_size].strip()
                if window:
                    chunks.append(
                        TextChunk(
                            chunk_id=str(uuid.uuid4()),
                            document_id=document_id,
                            text=window,
                            chunk_index=chunk_index,
                            filename=filename,
                            file_type=file_type,
                            document_hash=document_hash,
                            page_number=section.page_number,
                            slide_number=section.slide_number,
                            sheet_name=section.sheet_name,
                            section=section.section_title,
                            source=filename,
                        )
                    )
                    chunk_index += 1

        return chunks

    def _recursive_split_text(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Recursively splits string using standard separators."""
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._split_with_separators(text, max_size, overlap, separators)

    def _split_with_separators(
        self, text: str, max_size: int, overlap: int, separators: List[str]
    ) -> List[str]:
        if len(text) <= max_size or not separators:
            return [text] if text.strip() else []

        sep = separators[0]
        remaining_seps = separators[1:]
        splits = text.split(sep) if sep else list(text)

        chunks: List[str] = []
        current_part: List[str] = []
        current_len = 0

        for part in splits:
            part_len = len(part) + len(sep)
            if current_len + part_len > max_size and current_part:
                joined = sep.join(current_part)
                if len(joined) > max_size and remaining_seps:
                    chunks.extend(self._split_with_separators(joined, max_size, overlap, remaining_seps))
                else:
                    chunks.append(joined)

                # Overlap step
                current_part = [part]
                current_len = part_len
            else:
                current_part.append(part)
                current_len += part_len

        if current_part:
            joined = sep.join(current_part)
            if len(joined) > max_size and remaining_seps:
                chunks.extend(self._split_with_separators(joined, max_size, overlap, remaining_seps))
            else:
                chunks.append(joined)

        return chunks

    def _enrich_parent_child_context(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Assigns parent context windows to consecutive child chunks, bounded by page/slide/sheet borders."""
        if not chunks:
            return chunks

        def get_location_key(c: TextChunk):
            return (c.document_id, c.page_number, c.slide_number, c.sheet_name)

        parent_id = str(uuid.uuid4())
        current_parent_len = 0
        window_chunks: List[TextChunk] = []
        current_key = None

        for chunk in chunks:
            chunk_len = len(chunk.text)
            loc_key = get_location_key(chunk)

            boundary_changed = current_key is not None and loc_key != current_key
            size_exceeded = current_parent_len + chunk_len > self.parent_chunk_size

            if (boundary_changed or size_exceeded) and window_chunks:
                parent_text = "\n\n".join(c.text for c in window_chunks)
                for w_chunk in window_chunks:
                    w_chunk.parent_chunk_id = parent_id
                    w_chunk.parent_text = parent_text

                # Reset for next window
                parent_id = str(uuid.uuid4())
                window_chunks = [chunk]
                current_parent_len = chunk_len
                current_key = loc_key
            else:
                window_chunks.append(chunk)
                current_parent_len += chunk_len
                if current_key is None:
                    current_key = loc_key

        if window_chunks:
            parent_text = "\n\n".join(c.text for c in window_chunks)
            for w_chunk in window_chunks:
                w_chunk.parent_chunk_id = parent_id
                w_chunk.parent_text = parent_text

        return chunks
