"""Data models and abstractions for the retrieval pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedChunk:
    """Represents a candidate chunk retrieved and scored through the pipeline."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    retrieval_method: str  # 'semantic', 'keyword', 'hybrid', 'reranked'
    filename: str = ""
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    sheet_name: Optional[str] = None
    section: Optional[str] = None
    source: str = ""
    parent_chunk_id: Optional[str] = None
    parent_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_citation(self) -> str:
        """Formats the chunk into a standardized source attribution string."""
        parts = [f"Source: {self.filename or 'Document'}"]
        if self.page_number and self.page_number > 0:
            parts.append(f"Page: {self.page_number}")
        elif self.slide_number and self.slide_number > 0:
            parts.append(f"Slide: {self.slide_number}")
        elif self.sheet_name:
            parts.append(f"Sheet: {self.sheet_name}")
        elif self.section:
            parts.append(f"Section: {self.section}")
        return f"[{', '.join(parts)}]"
