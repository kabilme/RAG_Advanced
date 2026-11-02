"""Context compression and redundancy elimination engine."""

import re
from typing import List, Set
from app.core.config import settings
from app.core.logging import logger
from app.retrieval.base import RetrievedChunk


class ContextCompressor:
    """Compresses retrieved context by eliminating redundant text and selecting salient sentences."""

    _NUMERICAL_OR_FACT_REGEX = re.compile(
        r"(\d+(\.\d+)?%?|\b[A-Z][a-z]+\b|\b(is|are|defined|requires|measured|costs|leads to|results in)\b)",
        re.IGNORECASE
    )

    @classmethod
    def _jaccard_similarity(cls, tokens_a: Set[str], tokens_b: Set[str]) -> float:
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return intersection / union if union > 0 else 0.0

    @classmethod
    def deduplicate_chunks(cls, chunks: List[RetrievedChunk], threshold: float = 0.85) -> List[RetrievedChunk]:
        """Eliminates near-duplicate candidate chunks using token Jaccard similarity."""
        if len(chunks) <= 1:
            return chunks

        unique_chunks: List[RetrievedChunk] = []
        token_sets: List[Set[str]] = []

        for chunk in chunks:
            words = set(re.findall(r"\w+", chunk.text.lower()))
            is_dup = False
            for prev_tokens in token_sets:
                if cls._jaccard_similarity(words, prev_tokens) >= threshold:
                    is_dup = True
                    break

            if not is_dup:
                unique_chunks.append(chunk)
                token_sets.append(words)

        if len(unique_chunks) < len(chunks):
            logger.debug("Deduplicated chunks from %d to %d", len(chunks), len(unique_chunks))

        return unique_chunks

    @classmethod
    def compress_chunk_text(cls, query: str, text: str, max_chars: int = 1500) -> str:
        """Extracts the most query-salient sentences from a chunk, preserving facts and figures."""
        if len(text) <= max_chars:
            return text

        query_terms = set(re.findall(r"\w+", query.lower()))
        sentences = re.split(r"(?<=[.!?])\s+", text)

        scored_sentences = []
        for idx, sentence in enumerate(sentences):
            sent_words = set(re.findall(r"\w+", sentence.lower()))
            # Keyword overlap score
            overlap = len(sent_words.intersection(query_terms))
            # Has numbers or facts
            has_facts = bool(cls._NUMERICAL_OR_FACT_REGEX.search(sentence))
            score = (overlap * 2) + (1 if has_facts else 0)
            scored_sentences.append((idx, sentence, score))

        # Always keep highest scoring sentences
        scored_sentences.sort(key=lambda x: x[2], reverse=True)

        selected_indices = set()
        total_len = 0
        for idx, sent, _ in scored_sentences:
            if total_len + len(sent) > max_chars:
                break
            selected_indices.add(idx)
            total_len += len(sent)

        # Restore original chronological order of sentences
        ordered_sentences = [s for i, s, _ in sorted(scored_sentences, key=lambda x: x[0]) if i in selected_indices]
        compressed = " ".join(ordered_sentences)
        return compressed if compressed.strip() else text[:max_chars]

    @classmethod
    def compress(cls, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Runs deduplication and sentence compression across retrieved chunks."""
        if not settings.CONTEXT_COMPRESSION or not chunks:
            return chunks

        deduped = cls.deduplicate_chunks(chunks)
        compressed_chunks: List[RetrievedChunk] = []

        for chunk in deduped:
            compressed_text = cls.compress_chunk_text(query, chunk.text)
            # Create updated chunk copy
            compressed_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=compressed_text,
                    score=chunk.score,
                    rank=chunk.rank,
                    retrieval_method=chunk.retrieval_method,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    slide_number=chunk.slide_number,
                    sheet_name=chunk.sheet_name,
                    section=chunk.section,
                    source=chunk.source,
                    parent_chunk_id=chunk.parent_chunk_id,
                    parent_text=chunk.parent_text,
                    metadata=chunk.metadata,
                )
            )

        return compressed_chunks
