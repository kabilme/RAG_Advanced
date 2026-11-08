"""Cross-encoder reranking engine using Sentence Transformers."""

import math
from typing import List, Optional
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import logger
from app.retrieval.base import RetrievedChunk


class CrossEncoderReranker:
    """Re-scores and re-ranks retrieved candidate chunks using a local Cross-Encoder model."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        enabled: Optional[bool] = None,
        device: str = "cpu",
    ):
        self.model_name = model_name or settings.RERANKER_MODEL
        self.enabled = enabled if enabled is not None else settings.ENABLE_RERANKER
        self.device = device
        self._model: Optional[CrossEncoder] = None

    def _ensure_model(self) -> Optional[CrossEncoder]:
        """Lazy loads the CrossEncoder model."""
        if not self.enabled:
            return None

        if self._model is None:
            try:
                logger.info("Loading Cross-Encoder model '%s' on %s...", self.model_name, self.device)
                self._model = CrossEncoder(self.model_name, device=self.device)
                logger.info("Cross-Encoder '%s' successfully initialized", self.model_name)
            except Exception as exc:
                logger.warning(
                    "Failed to load Cross-Encoder '%s' (%s). Falling back gracefully to hybrid ranking.",
                    self.model_name, exc
                )
                self._model = None
        return self._model

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Converts raw cross-encoder logits into calibrated 0..1 probabilities."""
        try:
            return 1.0 / (1.0 + math.exp(-float(x)))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Reranks candidate chunks with cross-encoder relevance scoring.
        
        Args:
            query: User's original search query.
            candidates: Candidate chunks from initial retrieval.
            top_k: Number of highest-ranked chunks to retain.
            
        Returns:
            Re-sorted list of candidate chunks with updated relevance scores and ranks.
        """
        if not candidates or not query.strip():
            return candidates

        k = top_k or settings.RERANK_TOP_K

        model = self._ensure_model()
        if model is None:
            # Fall back directly to existing order if reranker is disabled or unavailable
            return candidates[:k]

        try:
            # Prepare query-document pairs
            pairs = [[query, chunk.text] for chunk in candidates]
            raw_scores = model.predict(pairs, show_progress_bar=False)

            # Pair candidates with their new cross-encoder scores
            scored_candidates = []
            for chunk, score in zip(candidates, raw_scores):
                calibrated_score = self._sigmoid(float(score))
                scored_candidates.append((chunk, calibrated_score))

            # Sort descending by cross-encoder score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_scored = scored_candidates[:k]

            reranked_results: List[RetrievedChunk] = []
            for rank, (chunk, score) in enumerate(top_scored, start=1):
                reranked_results.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=chunk.text,
                        score=score,
                        rank=rank,
                        retrieval_method="reranked",
                        filename=chunk.filename,
                        page_number=chunk.page_number,
                        slide_number=chunk.slide_number,
                        sheet_name=chunk.sheet_name,
                        section=chunk.section,
                        source=chunk.source,
                        parent_chunk_id=chunk.parent_chunk_id,
                        parent_text=chunk.parent_text,
                        context_prefix=chunk.context_prefix,
                        raw_text=chunk.raw_text,
                        metadata=chunk.metadata,
                    )
                )

            logger.debug(
                "Cross-Encoder reranked %d candidates -> retained top %d (best score: %.4f)",
                len(candidates), len(reranked_results), reranked_results[0].score if reranked_results else 0.0
            )
            return reranked_results

        except Exception as exc:
            logger.warning("Reranking encountered an error: %s. Using initial ranking.", exc)
            return candidates[:k]
