"""BM25 lexical keyword retrieval engine using rank_bm25."""

import re
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logging import logger
from app.ingestion.chunking import TextChunk
from app.retrieval.base import RetrievedChunk


class BM25SearchEngine:
    """Performs sparse keyword retrieval using the BM25Okapi algorithm."""

    def __init__(self):
        self._chunks: List[TextChunk] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Simple, fast lexical tokenizer stripping punctuation and lowercasing."""
        if not text:
            return []
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [word for word in cleaned.split() if len(word) > 1]

    def index_chunks(self, chunks: List[TextChunk]) -> None:
        """Indexes or appends chunks to the BM25 corpus."""
        if not chunks:
            return

        # Deduplicate against existing chunk_ids
        existing_ids = {c.chunk_id for c in self._chunks}
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            return

        self._chunks.extend(new_chunks)
        for chunk in new_chunks:
            self._tokenized_corpus.append(self.tokenize(chunk.text))

        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index updated: %d total indexed chunks", len(self._chunks))

    def remove_document(self, document_id: str) -> int:
        """Removes chunks belonging to a document and rebuilds the BM25 index."""
        initial_len = len(self._chunks)
        kept_chunks = []
        kept_tokens = []

        for chunk, tokens in zip(self._chunks, self._tokenized_corpus):
            if chunk.document_id != document_id:
                kept_chunks.append(chunk)
                kept_tokens.append(tokens)

        removed_count = initial_len - len(kept_chunks)
        if removed_count > 0:
            self._chunks = kept_chunks
            self._tokenized_corpus = kept_tokens
            self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
            logger.info("Removed %d chunks for document %s from BM25 index", removed_count, document_id)

        return removed_count

    def clear(self) -> None:
        """Clears the entire BM25 index."""
        self._chunks.clear()
        self._tokenized_corpus.clear()
        self._bm25 = None

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Retrieves top-K lexical matches using BM25 scoring."""
        if not self._bm25 or not self._chunks or not query.strip():
            return []

        k = top_k or settings.TOP_K
        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        doc_scores = self._bm25.get_scores(tokenized_query)

        # Match and filter
        scored_pairs = []
        for chunk, score in zip(self._chunks, doc_scores):
            if score <= 0.0:
                continue

            # Apply metadata filters if provided
            if filter_criteria:
                match = True
                for fk, fv in filter_criteria.items():
                    chunk_val = getattr(chunk, fk, None) or chunk.metadata.get(fk)
                    if chunk_val != fv:
                        match = False
                        break
                if not match:
                    continue

            scored_pairs.append((chunk, float(score)))

        if not scored_pairs:
            return []

        # Sort descending by BM25 score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = scored_pairs[:k]

        max_score = top_pairs[0][1] if top_pairs else 1.0

        candidates: List[RetrievedChunk] = []
        for rank, (chunk, score) in enumerate(top_pairs, start=1):
            # Normalize BM25 score to 0..1 range
            normalized_score = score / max_score if max_score > 0 else 0.0
            candidates.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=normalized_score,
                    rank=rank,
                    retrieval_method="keyword",
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    slide_number=chunk.slide_number,
                    sheet_name=chunk.sheet_name,
                    section=chunk.section,
                    source=chunk.source,
                    parent_chunk_id=chunk.parent_chunk_id,
                    parent_text=chunk.parent_text,
                    context_prefix=getattr(chunk, "context_prefix", None),
                    raw_text=getattr(chunk, "raw_text", None),
                    metadata=chunk.to_dict(),
                )
            )

        return candidates
