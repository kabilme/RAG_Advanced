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
    # [Source: example.pdf, Page 5]
    # [Source: notes.docx, Section: Intro]
    # [Source: presentation.pptx, Slide: 3]
    # [Source: data.xlsx, Sheet: Sales]
    # [Source: report.txt]
    _CITATION_REGEX = re.compile(
        r"\[Source:\s*([^,;\]]+)(?:[,;]\s*(Pages?|p\.|pp\.|Slides?|Sheets?|Sections?):?\s*([^\]]+))?\]",
        re.IGNORECASE,
    )

    @classmethod
    def normalize_location_type(cls, loc_type: Optional[str]) -> Optional[str]:
        """Normalizes location keywords (e.g. 'pages', 'p.', 'slides') to standard capitalized names."""
        if not loc_type:
            return None
        clean = loc_type.lower().rstrip(".:")
        if clean in ("page", "pages", "p", "pp"):
            return "Page"
        elif clean in ("slide", "slides"):
            return "Slide"
        elif clean in ("sheet", "sheets"):
            return "Sheet"
        elif clean in ("section", "sections"):
            return "Section"
        return loc_type.capitalize()

    @classmethod
    def normalize_citations(cls, text: str) -> str:
        """Normalizes various citation styles into standard format [Source: filename.ext, LocationType: Value]."""
        if not text:
            return ""

        # Normalize (filename.pdf, Page 18) or [filename.pdf, Page 18] or (Source: filename.pdf, Page: 18)
        def _repl_loc(match):
            fname = match.group(1).strip()
            loc_type = match.group(2).strip()
            loc_val = match.group(3).strip()
            norm_type = cls.normalize_location_type(loc_type) or "Page"
            return f"[Source: {fname}, {norm_type}: {loc_val}]"

        p1 = re.compile(
            r"[\(\[]\s*(?:Source:\s*)?([a-zA-Z0-9_\-\. ]+\.[a-zA-Z0-9]{2,5})\s*,\s*(Pages?|p\.|pp\.|Slides?|Sheets?|Sections?):?\s*([^\]\)]+)[\)\]]",
            re.IGNORECASE,
        )
        s = p1.sub(_repl_loc, text)

        # Normalize (filename.pdf) or [filename.pdf] without location
        p2 = re.compile(
            r"[\(\[]\s*(?:Source:\s*)?([a-zA-Z0-9_\-\. ]+\.[a-zA-Z0-9]{2,5})\s*[\)\]]",
            re.IGNORECASE,
        )
        def _repl_noloc(match):
            raw = match.group(0)
            if raw.startswith("[Source:"):
                return raw
            fname = match.group(1).strip()
            return f"[Source: {fname}]"

        s = p2.sub(_repl_noloc, s)
        return s

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
            loc_type_raw = match.group(2).strip() if match.group(2) else None
            loc_type = cls.normalize_location_type(loc_type_raw)
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
        4. Hallucinated page numbers (e.g., printed book pages) are auto-corrected
           to the actual retrieved chunk's location when the text context matches.
        """
        answer_text = cls.normalize_citations(answer_text)
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

            # Check for exact location match if specified
            matched_chunk = None
            if loc_type and loc_val:
                lt = loc_type.lower()
                for c in matching_chunks:
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

            if matched_chunk:
                # Valid citation with verified location
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
            elif loc_type and loc_val:
                # Location specified but no exact match in retrieved chunks.
                # Check if the surrounding text aligns with one of the retrieved chunks
                # (e.g. LLM used printed book page instead of PDF physical page index).
                citation_pos = answer_text.find(raw)
                window_start = max(0, citation_pos - 300)
                surrounding_text = answer_text[window_start:citation_pos].lower()
                surrounding_words = set(re.findall(r"\w{3,}", surrounding_text))

                best_chunk = None
                best_overlap = 0
                for c in matching_chunks:
                    chunk_words = set(re.findall(r"\w{3,}", c.text.lower()))
                    overlap = len(surrounding_words.intersection(chunk_words))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_chunk = c

                # Auto-correct if confident or single chunk from file
                if best_chunk and (best_overlap >= 3 or len(matching_chunks) == 1):
                    corrected_tag = best_chunk.to_citation()
                    logger.info(
                        "Auto-correcting citation for %s: '%s' -> '%s' (actual page: %s)",
                        fname, raw, corrected_tag, best_chunk.page_number
                    )
                    sanitized_text = sanitized_text.replace(raw, corrected_tag)
                    validated_items.append(
                        CitationItem(
                            raw_citation=corrected_tag,
                            filename=best_chunk.filename,
                            location_type="Page" if best_chunk.page_number else loc_type,
                            location_value=str(best_chunk.page_number) if best_chunk.page_number else loc_val,
                            is_valid=True,
                            document_id=best_chunk.document_id,
                            chunk_id=best_chunk.chunk_id,
                            snippet=best_chunk.text[:200] + ("..." if len(best_chunk.text) > 200 else ""),
                            metadata=best_chunk.metadata,
                        )
                    )
                else:
                    # Unverifiable hallucinated location
                    logger.warning("Stripping hallucinated location citation: %s", raw)
                    validated_items.append(
                        CitationItem(
                            raw_citation=raw,
                            filename=fname,
                            location_type=loc_type,
                            location_value=loc_val,
                            is_valid=False,
                        )
                    )
                    sanitized_text = sanitized_text.replace(raw, "")
            else:
                # Citation without location (e.g. [Source: doc.txt])
                fallback_chunk = matching_chunks[0]
                validated_items.append(
                    CitationItem(
                        raw_citation=raw,
                        filename=fallback_chunk.filename,
                        location_type=None,
                        location_value=None,
                        is_valid=True,
                        document_id=fallback_chunk.document_id,
                        chunk_id=fallback_chunk.chunk_id,
                        snippet=fallback_chunk.text[:200] + ("..." if len(fallback_chunk.text) > 200 else ""),
                        metadata=fallback_chunk.metadata,
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
                "context_prefix": chunk.context_prefix or "",
            })
        return cards
