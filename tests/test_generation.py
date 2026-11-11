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


def test_rag_chain_streaming_handles_heartbeat_pings(test_chunks):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = test_chunks

    mock_ollama = MagicMock()
    # Simulates heartbeats emitted while model evaluates prompt, followed by actual tokens
    mock_ollama.stream_generate.return_value = iter(["", "", "Solar ", "panels ", "last 25 years."])

    mock_memory = MagicMock()
    mock_memory.get_history.return_value = []

    chain = RAGChain(
        retrieval_pipeline=mock_retrieval,
        ollama_client=mock_ollama,
        conversation_manager=mock_memory,
    )

    events = list(chain.stream_query("Lifespan?", session_id="ping-test"))
    ping_events = [e for e in events if e.get("type") == "ping"]
    token_events = [e for e in events if e.get("type") == "token"]

    assert len(ping_events) == 2
    assert "".join(e["content"] for e in token_events) == "Solar panels last 25 years."


def test_ollama_client_acceleration_options():
    from app.core.config import settings

    client = OllamaClient(base_url="http://localhost:11434", model_name="llama3.2")

    # 1. When acceleration is disabled: must run unaccelerated with 1 thread and num_gpu=0
    with patch.object(settings, "ENABLE_ACCELERATION", False):
        opts = client._build_options()
        assert opts["num_thread"] == 1
        assert opts["num_gpu"] == 0

    # 2. When acceleration is manually enabled for CPU with 8 threads
    with patch.object(settings, "ENABLE_ACCELERATION", True), \
         patch.object(settings, "ACCELERATION_DEVICE", "cpu"), \
         patch.object(settings, "ACCELERATION_THREADS", 8):
        opts = client._build_options()
        assert opts["num_thread"] == 8
        assert opts["num_gpu"] == 0

    # 3. When acceleration is enabled for GPU
    with patch.object(settings, "ENABLE_ACCELERATION", True), \
         patch.object(settings, "ACCELERATION_DEVICE", "gpu"), \
         patch.object(settings, "ACCELERATION_THREADS", 4):
        opts = client._build_options()
        assert opts["num_gpu"] == 99


def test_clean_meta_language_removes_context_blocks():
    from app.generation.answer_validator import AnswerValidator

    raw_response = (
        "Based on the retrieved document context, some considerations to keep in mind in the production of goods for our current use include:\n\n"
        "• Reducing waste and pollution: As stated in Context Block 2, \"The factors - land, labour, capital, entrepreneurship and technology are combined to produce goods and services.\" (hees107.pdf, Page 18)\n"
        "• Using natural resources responsibly: Context Block 5 emphasizes the importance of using natural resources responsibly, stating, \"Businesses or Enterprises combine various inputs or factors of production to create goods and services.\" (hees107.pdf, Page 16)\n"
        "• Investing in technology and skills: Context Block 3 highlights the significance of technology and skills in production, noting, \"Technology is a facilitator.\" (hees107.pdf, Page 16)\n"
        "• Fair compensation and working conditions: Context Block 10 discusses the responsibilities of businesses towards their workers, stating, \"Fair compensation and working conditions: Employers need to ensure that workers are paid fairly for their labour.\" (hees107.pdf, Page 19)\n"
        "• Addressing social and environmental concerns: Context Block 10 highlights the importance of addressing social and environmental concerns, stating, \"Businesses are motivated to address social and environmental concerns.\" (hees107.pdf, Page 19)\n\n"
        "While the retrieved context does not provide a comprehensive list of considerations to keep in mind in the production of goods for our current use, it does offer valuable insights."
    )

    cleaned = AnswerValidator.clean_meta_language(raw_response)

    # 1. No "Context Block" should remain anywhere
    assert "Context Block" not in cleaned
    assert "context block" not in cleaned.lower()

    # 2. Robotic preamble should be stripped
    assert not cleaned.startswith("Based on the retrieved")
    assert cleaned.startswith("Some considerations to keep in mind")

    # 3. Repetitive inline titles should be deduplicated
    assert "Fair compensation and working conditions: Fair compensation and working conditions:" not in cleaned
    assert "Fair compensation and working conditions: Employers need to ensure" in cleaned

    # 4. Meta references in closing paragraph cleaned
    assert "the retrieved context" not in cleaned.lower()
    assert "the available documentation" in cleaned.lower()


def test_citation_normalization():
    from app.generation.citation_manager import CitationManager

    text_with_parens = "Here is a fact (hees107.pdf, Page 18) and another [report.docx, Section: Intro]."
    normalized = CitationManager.normalize_citations(text_with_parens)
    assert "[Source: hees107.pdf, Page: 18]" in normalized
    assert "[Source: report.docx, Section: Intro]" in normalized

    # Single document citation without location
    text_noloc = "As detailed in (notes.txt)."
    normalized_noloc = CitationManager.normalize_citations(text_noloc)
    assert "[Source: notes.txt]" in normalized_noloc


def test_prompt_builder_avoids_context_block_labels(test_chunks):
    context = PromptBuilder.format_context_block(test_chunks)
    assert "Context Block" not in context
    assert "Document Excerpt 1" in context


def test_prompt_builder_deduplicates_identical_parent_contexts():
    # 2 chunks with identical parent text (e.g. from the same page)
    parent_text = "This is a comprehensive parent section describing entire ecosystem services in detail."
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="doc1",
            text="First focused chunk about water.",
            score=0.9,
            rank=1,
            retrieval_method="hybrid",
            filename="ecosystems.pdf",
            page_number=6,
            parent_text=parent_text,
        ),
        RetrievedChunk(
            chunk_id="c2",
            document_id="doc1",
            text="Second focused chunk about soil.",
            score=0.85,
            rank=2,
            retrieval_method="hybrid",
            filename="ecosystems.pdf",
            page_number=6,
            parent_text=parent_text,
        ),
    ]

    context = PromptBuilder.format_context_block(chunks)
    # The full parent text should appear only once to save prompt tokens
    assert context.count(parent_text) == 1
    # The second excerpt should show its focused chunk text rather than repeating parent text
    assert "Second focused chunk about soil." in context


def test_format_professional_markdown_reconstructs_squashed_bullets():
    from app.generation.answer_validator import AnswerValidator

    squashed_raw = (
        "Here are five ecosystem functions that serve humans, supported by the retrieved document context: "
        "• Renewable Resource Management: Humans can manage renewable resources. [Source: hees101.pdf, Page: 6] "
        "• Fish Population Regulation: Traditional practices help maintain fish populations. [Source: hees101.pdf, Page: 6] "
        "• Water Security: The management of water resources is crucial. [Source: hees101.pdf, Page: 6] "
        "Citation Tags: hees101.pdf, Page 6 hees101.pdf, Page 6"
    )

    cleaned = AnswerValidator.clean_meta_language(squashed_raw)
    formatted = AnswerValidator.format_professional_markdown(cleaned)

    # 1. Trailing Citation Tags dump must be stripped
    assert "Citation Tags:" not in formatted

    # 2. Preambles must be cleaned
    assert "supported by the retrieved document context" not in formatted
    assert formatted.startswith("Here are five ecosystem functions that serve humans:")

    # 3. Bullets must not be inline - they must start with standard markdown list markers on newlines
    assert "•" not in formatted
    assert "\n\n- **Renewable Resource Management**: Humans can manage renewable resources." in formatted
    assert "\n\n- **Fish Population Regulation**: Traditional practices help maintain fish populations." in formatted
    assert "\n\n- **Water Security**: The management of water resources is crucial." in formatted

    # 4. Must render as distinct HTML list elements when parsed by markdown
    import markdown
    html = markdown.markdown(formatted)
    assert "<ul>" in html
    assert "<strong>Renewable Resource Management</strong>:" in html
    assert "<strong>Fish Population Regulation</strong>:" in html


def test_format_professional_markdown_aligns_first_item_and_promotes_title():
    from app.generation.answer_validator import AnswerValidator

    raw = (
        "Ecosystem Functions Serving Humans\n"
        "The retrieval document context provides several ecosystem functions that serve humans. "
        "Here are five examples: Filtering Water: ecosystems like forests naturally filter water. [Source: hees101.pdf, Page: 7]\n"
        "• Preventing Soil Erosion: Forests also prevent soil erosion. [Source: hees101.pdf, Page: 7]\n"
        "These ecosystem functions are vital for human survival."
    )

    cleaned = AnswerValidator.clean_meta_language(raw)
    formatted = AnswerValidator.format_professional_markdown(cleaned)

    # Title must be elevated to header ###
    assert formatted.startswith("### Ecosystem Functions Serving Humans\n\n")

    # Meta language cleaned
    assert "retrieval document context provides" not in formatted
    assert "The available documentation outlines" in formatted

    # First item must be detached into its own bullet with bold heading
    assert "\n\n- **Filtering Water**: ecosystems like forests naturally filter water." in formatted

    # Second item must be standard bullet
    assert "\n\n- **Preventing Soil Erosion**: Forests also prevent soil erosion." in formatted

    # Concluding paragraph must have a blank line before it
    assert "\n\nThese ecosystem functions are vital for human survival." in formatted




