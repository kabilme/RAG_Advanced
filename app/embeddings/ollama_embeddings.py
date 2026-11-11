"""Ollama-based embedding provider using local Ollama service."""

from typing import List, Optional
import ollama

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import (
    EmbeddingError,
    OllamaConnectionError,
    ModelNotAvailableError,
)
from app.embeddings.base import BaseEmbeddingProvider


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Generates dense vector embeddings using a locally hosted Ollama model."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 32,
    ):
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._base_url = base_url or settings.OLLAMA_BASE_URL
        self._batch_size = batch_size
        self._client = ollama.Client(host=self._base_url, timeout=getattr(settings, "TIMEOUT_SECONDS", 120))
        self._dimension: Optional[int] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = self._resolve_dimension()
        return self._dimension

    def check_health(self) -> bool:
        """Verifies connection to Ollama server and checks model availability."""
        try:
            models_response = self._client.list()
            # Extract names from ListResponse
            installed_models = []
            if hasattr(models_response, "models"):
                for m in models_response.models:
                    model_str = getattr(m, "model", None) or getattr(m, "name", "")
                    installed_models.append(model_str)
            elif isinstance(models_response, dict) and "models" in models_response:
                for m in models_response["models"]:
                    installed_models.append(m.get("name", m.get("model", "")))

            # Check if requested model or prefix matches
            target_prefix = self._model_name.split(":")[0]
            matched = any(
                m == self._model_name or m.startswith(f"{target_prefix}:")
                for m in installed_models
            )

            if not matched:
                logger.warning(
                    "Ollama is running, but embedding model '%s' was not found in installed models: %s",
                    self._model_name, installed_models
                )
                return False

            return True
        except Exception as exc:
            logger.warning("Failed to connect to Ollama at %s: %s", self._base_url, exc)
            return False

    def _resolve_dimension(self) -> int:
        """Determines vector dimension by running a single sample embedding."""
        try:
            sample_vec = self.embed_query("dimension_probe")
            dim = len(sample_vec)
            logger.info("Ollama embedding model '%s' vector dimension: %d", self._model_name, dim)
            return dim
        except Exception as exc:
            logger.warning("Could not automatically probe dimension for '%s': %s", self._model_name, exc)
            # Default fallback for popular models like nomic-embed-text (768) or bge-small (384)
            return 768

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of document chunks."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        try:
            # Process in configurable batches
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]

                # Try batch embed API
                try:
                    response = self._client.embed(model=self._model_name, input=batch)
                    if hasattr(response, "embeddings") and response.embeddings:
                        all_embeddings.extend(response.embeddings)
                    elif isinstance(response, dict) and "embeddings" in response:
                        all_embeddings.extend(response["embeddings"])
                    else:
                        raise ValueError("Unexpected response structure from ollama.embed")
                except Exception as batch_err:
                    logger.debug("Batch embed fallback to single embed calls: %s", batch_err)
                    for text in batch:
                        resp = self._client.embeddings(model=self._model_name, prompt=text)
                        embedding = resp.get("embedding") if isinstance(resp, dict) else getattr(resp, "embedding", None)
                        if not embedding:
                            raise EmbeddingError(f"Ollama returned empty embedding for text: {text[:40]}")
                        all_embeddings.append(embedding)

        except Exception as exc:
            if "connection" in str(exc).lower() or "connect" in str(exc).lower():
                raise OllamaConnectionError(f"Cannot connect to Ollama at {self._base_url}: {exc}") from exc
            raise EmbeddingError(f"Ollama embedding generation failed: {exc}") from exc

        # Set dimension if not yet determined
        if self._dimension is None and all_embeddings:
            self._dimension = len(all_embeddings[0])

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generates embedding for a single query."""
        if not query.strip():
            raise EmbeddingError("Cannot generate embedding for empty query.")

        try:
            response = self._client.embed(model=self._model_name, input=query)
            if hasattr(response, "embeddings") and response.embeddings:
                return response.embeddings[0]
            elif isinstance(response, dict) and "embeddings" in response:
                return response["embeddings"][0]

            # Fallback to single embedding endpoint
            resp = self._client.embeddings(model=self._model_name, prompt=query)
            return resp.get("embedding") if isinstance(resp, dict) else getattr(resp, "embedding", [])
        except Exception as exc:
            if "connection" in str(exc).lower() or "connect" in str(exc).lower():
                raise OllamaConnectionError(f"Cannot connect to Ollama at {self._base_url}: {exc}") from exc
            raise EmbeddingError(f"Ollama query embedding failed for model '{self._model_name}': {exc}") from exc
