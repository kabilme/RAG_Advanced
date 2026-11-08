"""Unit and integration tests for Contextual Retrieval chunk enrichment."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.ingestion.contextual_enricher import ContextualEnricher
from app.ingestion.loaders import DocumentSection
from app.ingestion.chunking import DocumentChunker, TextChunk
from app.retrieval.keyword_search import BM25SearchEngine


def test_clean_document_title():
    assert ContextualEnricher.clean_document_title("Soft_Skills_III_Unit_I_MS_Excel_Basics.docx") == "Soft Skills III Unit I MS Excel Basics"
    assert ContextualEnricher.clean_document_title("annual-financial-report_2024.pdf") == "annual financial report 2024"
    assert ContextualEnricher.clean_document_title("simple_notes.txt") == "simple notes"


def test_generate_structural_prefix():
    prefix_pdf = ContextualEnricher.generate_structural_prefix(
        filename="User_Guide.pdf",
        section_title="Authentication & Security",
        page_number=4,
    )
    assert "[Document: User_Guide.pdf | Section: Authentication & Security | Page: 4]" in prefix_pdf
    assert "Context: User Guide — Authentication & Security." in prefix_pdf

    prefix_pptx = ContextualEnricher.generate_structural_prefix(
        filename="Pitch_Deck.pptx",
        slide_number=7,
    )
    assert "[Document: Pitch_Deck.pptx | Slide: 7]" in prefix_pptx
    assert "Context: Pitch Deck (Slide 7)." in prefix_pptx

    prefix_xlsx = ContextualEnricher.generate_structural_prefix(
        filename="Financials.xlsx",
        sheet_name="Q4_Forecast",
    )
    assert "[Document: Financials.xlsx | Sheet: Q4_Forecast]" in prefix_xlsx
    assert "Context: Financials (Sheet 'Q4_Forecast')." in prefix_xlsx


def test_generate_llm_prefix_with_mock():
    mock_ollama = MagicMock()
    mock_ollama.generate.return_value = {
        "response": "This section outlines OAuth2 bearer token authentication protocols used by backend services."
    }

    llm_prefix = ContextualEnricher.generate_llm_prefix(
        chunk_text="Clients must provide an Authorization header with Bearer token.",
        filename="Security_Architecture.pdf",
        section_title="API Auth",
        page_number=12,
        ollama_client=mock_ollama,
    )
    assert "[Document: Security_Architecture.pdf | Section: API Auth | Page: 12]" in llm_prefix
    assert "OAuth2 bearer token authentication protocols" in llm_prefix

    # Test fallback on exception
    mock_failing_ollama = MagicMock()
    mock_failing_ollama.generate.side_effect = RuntimeError("Ollama connection timed out")
    fallback_prefix = ContextualEnricher.generate_llm_prefix(
        chunk_text="Some text",
        filename="Security_Architecture.pdf",
        section_title="API Auth",
        page_number=12,
        ollama_client=mock_failing_ollama,
    )
    assert "[Document: Security_Architecture.pdf | Section: API Auth | Page: 12]" in fallback_prefix
    assert "Context: Security Architecture — API Auth." in fallback_prefix


def test_enrich_chunks_caching_and_text_structure():
    chunks = [
        TextChunk(
            chunk_id="c1",
            document_id="doc1",
            text="First sentence of chapter.",
            chunk_index=0,
            filename="Handbook.docx",
            file_type="docx",
            document_hash="hash1",
            section="Onboarding",
            page_number=1,
        ),
        TextChunk(
            chunk_id="c2",
            document_id="doc1",
            text="Second sentence of chapter.",
            chunk_index=1,
            filename="Handbook.docx",
            file_type="docx",
            document_hash="hash1",
            section="Onboarding",
            page_number=1,
        ),
    ]

    enriched = ContextualEnricher.enrich_chunks(chunks, filename="Handbook.docx", mode="structural")
    assert len(enriched) == 2

    for c in enriched:
        assert c.context_prefix is not None
        assert "Handbook.docx" in c.context_prefix
        assert "Onboarding" in c.context_prefix
        assert c.raw_text in ("First sentence of chapter.", "Second sentence of chapter.")
        assert c.text.startswith(c.context_prefix)
        assert c.raw_text in c.text

    # Both chunks in same section/page share identical context prefix
    assert enriched[0].context_prefix == enriched[1].context_prefix


def test_bm25_matches_document_and_section_keywords_from_context():
    """Validates the core value proposition of Contextual Retrieval:
    BM25 finds chunks matching document and section keywords that were absent in raw text.
    """
    raw_body = "The function takes three arguments: range, criteria, and sum_range."
    chunk = TextChunk(
        chunk_id="chunk-excel-1",
        document_id="doc-excel",
        text=raw_body,
        chunk_index=0,
        filename="Soft_Skills_III_Unit_I_MS_Excel_Basics.docx",
        file_type="docx",
        document_hash="hash-excel",
        section="Mathematical Formulas",
        page_number=5,
    )

    # Before contextual enrichment: raw text does NOT contain 'Excel' or 'Soft Skills' or 'Mathematical'
    assert "excel" not in raw_body.lower()
    assert "soft skills" not in raw_body.lower()

    # Apply contextual enrichment
    ContextualEnricher.enrich_chunks([chunk], filename="Soft_Skills_III_Unit_I_MS_Excel_Basics.docx")

    # Add unrelated background documents to provide realistic multi-document IDF in BM25 (N >= 3)
    chunk2 = TextChunk(
        chunk_id="chunk-astronomy-2",
        document_id="doc-astronomy",
        text="Planets revolve around the sun in elliptical orbits.",
        chunk_index=0,
        filename="Astronomy.pdf",
        file_type="pdf",
        document_hash="hash-astro",
        section="Solar System",
        page_number=1,
    )
    ContextualEnricher.enrich_chunks([chunk2], filename="Astronomy.pdf")

    chunk3 = TextChunk(
        chunk_id="chunk-biology-3",
        document_id="doc-biology",
        text="Mitochondria are the powerhouses of eukaryotic cells.",
        chunk_index=0,
        filename="Cell_Biology.txt",
        file_type="txt",
        document_hash="hash-bio",
        section="Cell Organelles",
    )
    ContextualEnricher.enrich_chunks([chunk3], filename="Cell_Biology.txt")

    # Index into BM25
    engine = BM25SearchEngine()
    engine.index_chunks([chunk, chunk2, chunk3])

    # Search for keywords that only exist in the prepended context
    results = engine.search(query="Excel mathematical formula")
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-excel-1"
    assert results[0].score > 0.0

    # Search for course keywords
    course_results = engine.search(query="Soft Skills Unit 1")
    assert len(course_results) == 1
    assert course_results[0].chunk_id == "chunk-excel-1"
