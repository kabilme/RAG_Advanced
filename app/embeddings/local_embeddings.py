"""Local SentenceTransformers embedding provider."""

from typing import List, Optional
from sentence_transformers import SentenceTransformer

from app.core.logging import logger
from app.core.exceptions import EmbeddingError
from app.embeddings.base import BaseEmbeddingProvider


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Generates dense vector embeddings locally using Sentence Transformers on CPU/GPU."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None

    def _ensure_model(self) -> SentenceTransformer:
        """Loads SentenceTransformer model on demand."""
        if self._model is None:
            try:
                logger.info("Loading SentenceTransformer model '%s' on %s...", self._model_name, self._device)
                self._model = SentenceTransformer(self._model_name, device=self._device)
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = self._model.get_embedding_dimension()
                else:
                    self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info("SentenceTransformer '%s' ready (dimension: %d)", self._model_name, self._dimension)
            except Exception as exc:
                raise EmbeddingError(f"Failed to load SentenceTransformer model '{self._model_name}': {exc}") from exc
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_model()
        return self._dimension  # type: ignore

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of document strings."""
        if not texts:
            return []

        model = self._ensure_model()
        try:
            embeddings = model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Local embedding generation failed: {exc}") from exc

    def embed_query(self, query: str) -> List[float]:
        """Generates embedding for a search query."""
        if not query.strip():
            raise EmbeddingError("Cannot embed empty query.")

        model = self._ensure_model()
        try:
            vec = model.encode(
                query,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vec.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Local query embedding failed: {exc}") from exc
