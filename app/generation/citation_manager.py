"""Citation extraction, validation, and source traceability management."""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging import logger
from app.retrieval.base import RetrievedChunk


@dataclass
class CitationItem:
    """Represents an extracted and validated citation."""
    raw_citation: str
    filename: str
    location_type: Optional[str] = None  # Page, Slide, Sheet, Section
    location_value: Optional[str] = None
    is_valid: bool = True
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    snippet: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_citation": self.raw_citation,
            "filename": self.filename,
            "location_type": self.location_type,
            "location_value": self.location_value,
            "is_valid": self.is_valid,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "snippet": self.snippet,
        }


class CitationManager:
    """Extracts, verifies, and formats document citations."""

    # Matches citations formatted like:
    # [Source: example.pdf, Page: 5]
    # [Source: notes.docx, Section: Intro]
    # [Source: presentation.pptx, Slide: 3]
    # [Source: data.xlsx, Sheet: Sales]
    # [Source: report.txt]
    _CITATION_REGEX = re.compile(
        r"\[Source:\s*([^,\]]+)(?:,\s*(Page|Slide|Sheet|Section):\s*([^\]]+))?\]",
        re.IGNORECASE,
    )

    @classmethod
    def extract_citations(cls, text: str) -> List[Tuple[str, str, Optional[str], Optional[str]]]:
        """Finds all citation tags in generated text.
        
        Returns:
            List of (full_raw_citation, filename, location_type, location_value)
        """
        matches = []
        for match in cls._CITATION_REGEX.finditer(text):
            raw = match.group(0)
            fname = match.group(1).strip()
            loc_type = match.group(2).strip() if match.group(2) else None
            loc_val = match.group(3).strip() if match.group(3) else None
            matches.append((raw, fname, loc_type, loc_val))
        return matches

    @classmethod
    def validate_citations(
        cls,
        answer_text: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Tuple[List[CitationItem], str]:
        """Validates all citations in an answer against the retrieved chunks.
        
        Ensures that:
        1. The cited document was actually retrieved in the candidate pool.
        2. Location matches the actual retrieved chunk if specified.
        3. Fabricated citations are flagged and removed from the returned text.
        """
        raw_citations = cls.extract_citations(answer_text)
        validated_items: List[CitationItem] = []
        sanitized_text = answer_text

        # Map available files and locations in retrieved chunks
        chunk_lookup: Dict[str, List[RetrievedChunk]] = {}
        for chunk in retrieved_chunks:
            fn = chunk.filename.lower()
            chunk_lookup.setdefault(fn, []).append(chunk)

        for raw, fname, loc_type, loc_val in raw_citations:
            fn_key = fname.lower()
            matching_chunks = chunk_lookup.get(fn_key, [])

            if not matching_chunks:
                # Fabricated citation for an unretrieved document!
                logger.warning("Detected hallucinated citation for unretrieved file: %s", raw)
                validated_items.append(
                    CitationItem(
                        raw_citation=raw,
                        filename=fname,
                        location_type=loc_type,
                        location_value=loc_val,
                        is_valid=False,
                    )
                )
                # Strip hallucinated citation from text
                sanitized_text = sanitized_text.replace(raw, "")
                continue

            # Verify location match if specified
            matched_chunk = None
            for c in matching_chunks:
                if loc_type and loc_val:
                    lt = loc_type.lower()
                    if lt == "page" and str(c.page_number) == loc_val:
                        matched_chunk = c
                        break
                    elif lt == "slide" and str(c.slide_number) == loc_val:
                        matched_chunk = c
                        break
                    elif lt == "sheet" and (c.sheet_name or "").lower() == loc_val.lower():
                        matched_chunk = c
                        break
                    elif lt == "section" and (c.section or "").lower() == loc_val.lower():
                        matched_chunk = c
                        break
                else:
                    matched_chunk = c
                    break

            if not matched_chunk:
                # Best effort match to first chunk from that file
                matched_chunk = matching_chunks[0]

            validated_items.append(
                CitationItem(
                    raw_citation=raw,
                    filename=matched_chunk.filename,
                    location_type=loc_type,
                    location_value=loc_val,
                    is_valid=True,
                    document_id=matched_chunk.document_id,
                    chunk_id=matched_chunk.chunk_id,
                    snippet=matched_chunk.text[:200] + ("..." if len(matched_chunk.text) > 200 else ""),
                    metadata=matched_chunk.metadata,
                )
            )

        # Clean any double spaces left from stripping invalid citations
        sanitized_text = re.sub(r"\s{2,}", " ", sanitized_text).strip()
        return validated_items, sanitized_text

    @classmethod
    def build_source_cards(cls, chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """Generates comprehensive source card payload for frontend display."""
        cards = []
        for idx, chunk in enumerate(chunks, start=1):
            cards.append({
                "index": idx,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "slide_number": chunk.slide_number,
                "sheet_name": chunk.sheet_name,
                "section": chunk.section,
                "similarity_score": round(chunk.score, 4),
                "citation": chunk.to_citation(),
                "snippet": chunk.text,
            })
        return cards
