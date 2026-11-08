"""Contextual Retrieval: Enrich chunks with situated document and section preambles."""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.ingestion.loaders import DocumentSection


class ContextualEnricher:
    """Enriches discrete text chunks with situated document and section preambles.
    
    Implements Contextual Retrieval (Anthropic-style) to prevent 'amnesiac chunk'
    retrieval failures by prepending situational context before
    embedding generation and BM25 tokenization.
    """

    @classmethod
    def clean_document_title(cls, filename: str) -> str:
        """Derives a human-readable title from a filename by stripping extension and normalizing punctuation."""
        stem = Path(filename).stem
        # Replace underscores and hyphens with spaces
        clean = re.sub(r"[_\-]+", " ", stem)
        # Collapse multiple spaces
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @classmethod
    def generate_structural_prefix(
        cls,
        filename: str,
        section_title: Optional[str] = None,
        page_number: Optional[int] = None,
        slide_number: Optional[int] = None,
        sheet_name: Optional[str] = None,
    ) -> str:
        """Constructs a deterministic, high-information contextual preamble from document structure.
        
        Zero extra latency; 100% offline and deterministic. Strictly scoped to the chunk's
        own location to prevent cross-page or cross-section contamination.
        """
        clean_title = cls.clean_document_title(filename)

        # Location details in header tag
        loc_parts = [f"Document: {filename}"]
        if section_title and section_title.strip():
            loc_parts.append(f"Section: {section_title.strip()}")
        if page_number and page_number > 0:
            loc_parts.append(f"Page: {page_number}")
        elif slide_number and slide_number > 0:
            loc_parts.append(f"Slide: {slide_number}")
        elif sheet_name and sheet_name.strip():
            loc_parts.append(f"Sheet: {sheet_name.strip()}")

        header_line = f"[{' | '.join(loc_parts)}]"

        # Contextual summary line
        if section_title and section_title.strip():
            topic_desc = f"Context: {clean_title} — {section_title.strip()}."
        else:
            loc_desc = []
            if page_number and page_number > 0:
                loc_desc.append(f"Page {page_number}")
            elif slide_number and slide_number > 0:
                loc_desc.append(f"Slide {slide_number}")
            elif sheet_name and sheet_name.strip():
                loc_desc.append(f"Sheet '{sheet_name.strip()}'")

            if loc_desc:
                topic_desc = f"Context: {clean_title} ({', '.join(loc_desc)})."
            else:
                topic_desc = f"Context: {clean_title}."

        return f"{header_line}\n{topic_desc}"

    @classmethod
    def generate_llm_prefix(
        cls,
        chunk_text: str,
        filename: str,
        section_title: Optional[str] = None,
        page_number: Optional[int] = None,
        slide_number: Optional[int] = None,
        sheet_name: Optional[str] = None,
        ollama_client: Optional[Any] = None,
    ) -> str:
        """Generates dynamic situational context using local Ollama model with structural fallback."""
        fallback = cls.generate_structural_prefix(
            filename=filename,
            section_title=section_title,
            page_number=page_number,
            slide_number=slide_number,
            sheet_name=sheet_name,
        )

        if not ollama_client:
            return fallback

        clean_title = cls.clean_document_title(filename)
        prompt = (
            f"You are a document indexing assistant. Situate the chunk within the document.\n"
            f"Document Title: {clean_title}\n"
            f"Section: {section_title or 'General'}\n"
            f"Location: Page {page_number if page_number else 'N/A'}\n\n"
            f"Chunk Text:\n{chunk_text[:500]}\n\n"
            f"Instruction: Output exactly 1 to 2 sentences describing what this chunk discusses and which document it belongs to. "
            f"Do not write 'Here is' or 'This chunk'. Be direct and concise."
        )

        try:
            resp = ollama_client.generate(
                prompt=prompt,
                system_prompt="You are an expert technical indexer. Write concise 1-2 sentence contextual descriptions for retrieval.",
                temperature=0.1,
                max_tokens=80,
            )
            llm_text = resp.get("response", "").strip() if isinstance(resp, dict) else str(resp).strip()
            if llm_text and len(llm_text) > 20:
                loc_parts = [f"Document: {filename}"]
                if section_title and section_title.strip():
                    loc_parts.append(f"Section: {section_title.strip()}")
                if page_number and page_number > 0:
                    loc_parts.append(f"Page: {page_number}")
                header = f"[{' | '.join(loc_parts)}]"
                return f"{header}\nContext: {llm_text}"
        except Exception as exc:
            logger.debug("Ollama contextual generation failed, falling back to structural: %s", exc)

        return fallback

    @classmethod
    def enrich_chunks(
        cls,
        chunks: List[Any],
        filename: str,
        sections: Optional[List[DocumentSection]] = None,
        mode: str = "structural",
        ollama_client: Optional[Any] = None,
    ) -> List[Any]:
        """Prepends contextual situational preambles to all text chunks in the document.
        
        Args:
            chunks: List of TextChunk objects to enrich.
            filename: The source file name.
            sections: Original extracted document sections.
            mode: 'structural' (deterministic, instant) or 'llm' (local LLM synthesis).
            ollama_client: Optional Ollama client for LLM mode.
            
        Returns:
            The list of enriched TextChunk objects with updated text and metadata.
        """
        if not chunks:
            return chunks

        section_context_cache: Dict[str, str] = {}

        for chunk in chunks:
            sec_name = chunk.section or "main"
            cache_key = f"{sec_name}_{chunk.page_number}_{chunk.slide_number}_{chunk.sheet_name}"

            if cache_key in section_context_cache:
                prefix = section_context_cache[cache_key]
            else:
                if mode == "llm" and ollama_client:
                    prefix = cls.generate_llm_prefix(
                        chunk_text=chunk.text,
                        filename=filename,
                        section_title=chunk.section,
                        page_number=chunk.page_number,
                        slide_number=chunk.slide_number,
                        sheet_name=chunk.sheet_name,
                        ollama_client=ollama_client,
                    )
                else:
                    prefix = cls.generate_structural_prefix(
                        filename=filename,
                        section_title=chunk.section,
                        page_number=chunk.page_number,
                        slide_number=chunk.slide_number,
                        sheet_name=chunk.sheet_name,
                    )
                section_context_cache[cache_key] = prefix

            # Preserve raw text and attach prefix
            chunk.context_prefix = prefix
            chunk.raw_text = chunk.text
            chunk.text = f"{prefix}\n\n{chunk.raw_text}"

            # Also enrich parent text if parent-child context is present
            if chunk.parent_text:
                chunk.parent_text = f"{prefix}\n\n{chunk.parent_text}"

        logger.info(
            "Contextual Retrieval enriched %d chunks for '%s' using mode '%s'",
            len(chunks), filename, mode
        )
        return chunks
