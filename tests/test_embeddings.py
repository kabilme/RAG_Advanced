"""Unit tests for embedding providers and embedding service."""

from unittest.mock import MagicMock, patch
import pytest

from app.embeddings.base import BaseEmbeddingProvider
from app.embeddings.local_embeddings import SentenceTransformerEmbeddingProvider
from app.embeddings.ollama_embeddings import OllamaEmbeddingProvider
from app.embeddings.embedding_service import EmbeddingService
from app.core.exceptions import EmbeddingError, OllamaConnectionError


def test_sentence_transformer_provider():
    # Test local CPU embedding generation
    provider = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    assert provider.dimension == 384
    assert provider.model_name == "all-MiniLM-L6-v2"

    texts = ["Retrieval-Augmented Generation", "Vector embeddings in high dimensions"]
    embeddings = provider.embed_documents(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384

    query_vec = provider.embed_query("What is RAG?")
    assert len(query_vec) == 384


def test_embedding_service_caching_and_stats():
    # Use SentenceTransformer under the hood
    service = EmbeddingService(provider_type="sentence_transformers", model_name="all-MiniLM-L6-v2")

    assert service.dimension == 384

    # Embed query first time -> cache miss
    vec1 = service.embed_query("Local AI Assistant")
    stats1 = service.get_stats()
    assert stats1["queries_embedded"] == 1
    assert stats1["cache_hits"] == 0

    # Embed same query second time -> cache hit!
    vec2 = service.embed_query("Local AI Assistant")
    stats2 = service.get_stats()
    assert stats2["queries_embedded"] == 1
    assert stats2["cache_hits"] == 1
    assert vec1 == vec2


def test_ollama_provider_mock():
    with patch("ollama.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Mock list response
        mock_client.list.return_value = {"models": [{"name": "nomic-embed-text:latest"}]}

        # Mock embed response
        mock_client.embed.return_value = {"embeddings": [[0.1] * 768]}

        provider = OllamaEmbeddingProvider(model_name="nomic-embed-text")
        assert provider.check_health() is True

        vec = provider.embed_query("test query")
        assert len(vec) == 768
        assert provider.dimension == 768


def test_ollama_connection_error():
    with patch("ollama.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.embed.side_effect = Exception("Connection refused to 127.0.0.1:11434")

        provider = OllamaEmbeddingProvider(model_name="nomic-embed-text")
        with pytest.raises(OllamaConnectionError):
            provider.embed_query("test")
