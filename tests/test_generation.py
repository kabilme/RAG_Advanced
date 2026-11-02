"""Unit tests for prompt construction, Ollama client, and full RAG chain."""

from unittest.mock import MagicMock, patch
import pytest

from app.retrieval.base import RetrievedChunk
from app.generation.prompt_builder import PromptBuilder, RAG_SYSTEM_PROMPT
from app.generation.ollama_client import OllamaClient
from app.generation.rag_chain import RAGChain
from app.generation.answer_validator import ABSTENTION_MESSAGE


@pytest.fixture
def test_chunks():
    return [
        RetrievedChunk(
            chunk_id="chunk-sol-1",
            document_id="doc-sol",
            text="Solar panels have an expected lifespan of 25 to 30 years.",
            score=0.91,
            rank=1,
            retrieval_method="hybrid",
            filename="solar_lifecycle.pdf",
            page_number=3,
        )
    ]


def test_prompt_builder_structure(test_chunks):
    history = [
        {"role": "user", "content": "What is the lifespan of solar panels?"},
        {"role": "assistant", "content": "Solar panels last between 25 and 30 years."},
    ]
    prompt = PromptBuilder.build_user_prompt(
        question="How long do they last?",
        chunks=test_chunks,
        conversation_history=history,
    )

    assert "RETRIEVED DOCUMENT CONTEXT" in prompt
    assert "[Source: solar_lifecycle.pdf, Page: 3]" in prompt
    assert "Solar panels have an expected lifespan of 25 to 30 years." in prompt
    assert "USER QUESTION:" in prompt
    assert "How long do they last?" in prompt
    assert "RECENT CONVERSATION HISTORY" in prompt


def test_rag_chain_successful_grounded_answer(test_chunks):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = test_chunks

    mock_ollama = MagicMock()
    mock_ollama.generate.return_value = (
        "Solar panels typically last 25 to 30 years [Source: solar_lifecycle.pdf, Page: 3]."
    )

    mock_memory = MagicMock()
    mock_memory.get_history.return_value = []

    chain = RAGChain(
        retrieval_pipeline=mock_retrieval,
        ollama_client=mock_ollama,
        conversation_manager=mock_memory,
    )

    resp = chain.query("What is the lifespan of solar panels?", session_id="test-session")

    assert resp.is_grounded is True
    assert resp.is_abstained is False
    assert "25 to 30 years" in resp.answer
    assert len(resp.citations) == 1
    assert resp.citations[0]["filename"] == "solar_lifecycle.pdf"
    assert resp.citations[0]["location_value"] == "3"

    # Verify turn persisted to memory
    mock_memory.add_user_message.assert_called_once()
    mock_memory.add_assistant_message.assert_called_once()


def test_rag_chain_insufficient_context_abstains():
    mock_retrieval = MagicMock()
    # Retrieval returns empty candidate list
    mock_retrieval.retrieve.return_value = []

    mock_ollama = MagicMock()
    mock_memory = MagicMock()
    mock_memory.get_history.return_value = []

    chain = RAGChain(
        retrieval_pipeline=mock_retrieval,
        ollama_client=mock_ollama,
        conversation_manager=mock_memory,
    )

    resp = chain.query("What is the King of Ujjain's favorite horse?", session_id="test-session")

    assert resp.is_abstained is True
    assert resp.answer == ABSTENTION_MESSAGE
    # Ollama generation should NOT even be called when context is empty!
    mock_ollama.generate.assert_not_called()


def test_rag_chain_streaming(test_chunks):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = test_chunks

    mock_ollama = MagicMock()
    mock_ollama.stream_generate.return_value = iter(["Solar ", "panels ", "last ", "25 to 30 years [Source: solar_lifecycle.pdf, Page: 3]."])

    mock_memory = MagicMock()
    mock_memory.get_history.return_value = []

    chain = RAGChain(
        retrieval_pipeline=mock_retrieval,
        ollama_client=mock_ollama,
        conversation_manager=mock_memory,
    )

    events = list(chain.stream_query("How long do solar panels last?", session_id="stream-sess"))

    assert len(events) >= 3
    # First event should be source cards
    assert events[0]["type"] == "sources"
    assert len(events[0]["data"]) == 1

    # Token events
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert "".join(tokens) == "Solar panels last 25 to 30 years [Source: solar_lifecycle.pdf, Page: 3]."

    # Final event
    done_event = events[-1]
    assert done_event["type"] == "done"
    assert done_event["is_grounded"] is True
    assert len(done_event["citations"]) == 1
