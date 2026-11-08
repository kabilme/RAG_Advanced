"""Unit tests for intelligent document chunking strategies."""

import pytest
from app.ingestion.chunking import DocumentChunker, TextChunk
from app.ingestion.loaders import DocumentSection


def test_structure_aware_chunking():
    chunker = DocumentChunker(
        strategy="structure_aware",
        chunk_size=100,
        chunk_overlap=20,
        enable_parent_child=False
    )
    sections = [
        DocumentSection(text="Short section.", section_title="Intro", page_number=1),
        DocumentSection(
            text="This is a much longer section that will definitely exceed the small chunk size of 100 characters so that it tests recursive splitting behavior correctly across multiple chunks.",
            section_title="Deep Dive",
            page_number=2
        ),
    ]

    chunks = chunker.chunk_document(
        document_id="doc-123",
        filename="test.pdf",
        file_type="pdf",
        document_hash="hash-123",
        sections=sections
    )

    assert len(chunks) >= 3
    # Check section title preservation
    assert chunks[0].section == "Intro"
    assert chunks[0].page_number == 1
    assert chunks[1].section == "Deep Dive"
    assert chunks[1].page_number == 2


def test_sentence_chunking():
    chunker = DocumentChunker(
        strategy="sentence",
        chunk_size=60,
        chunk_overlap=15,
        enable_parent_child=False
    )
    sections = [
        DocumentSection(
            text="First sentence here. Second sentence follows. Third sentence wraps up this test block.",
            section_title="Sentences"
        )
    ]
    chunks = chunker.chunk_document(
        document_id="doc-456",
        filename="notes.txt",
        file_type="txt",
        document_hash="hash-456",
        sections=sections
    )

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.text) > 0


def test_paragraph_chunking():
    chunker = DocumentChunker(
        strategy="paragraph",
        enable_parent_child=False,
        enable_contextual=False,
    )
    sections = [
        DocumentSection(
            text="Paragraph one content.\n\nParagraph two content.\n\nParagraph three content.",
            section_title="Article"
        )
    ]
    chunks = chunker.chunk_document(
        document_id="doc-789",
        filename="article.txt",
        file_type="txt",
        document_hash="hash-789",
        sections=sections
    )

    assert len(chunks) == 3
    assert chunks[0].text == "Paragraph one content."
    assert chunks[1].text == "Paragraph two content."
    assert chunks[2].text == "Paragraph three content."


def test_contextual_retrieval_enrichment():
    """Verifies that Contextual Retrieval prepends situated document and section preambles."""
    chunker = DocumentChunker(
        strategy="structure_aware",
        enable_parent_child=True,
        enable_contextual=True,
        contextual_mode="structural",
    )
    sections = [
        DocumentSection(
            text="SUMIF adds all numbers in a range of cells based on one criterion.",
            section_title="Statistical Functions",
            page_number=3,
        )
    ]
    chunks = chunker.chunk_document(
        document_id="doc-ctx-1",
        filename="Soft_Skills_III_Unit_I_MS_Excel_Basics.docx",
        file_type="docx",
        document_hash="hash-ctx-1",
        sections=sections,
    )

    assert len(chunks) == 1
    c = chunks[0]
    # Check that raw text is preserved
    assert c.raw_text == "SUMIF adds all numbers in a range of cells based on one criterion."
    # Check that context prefix is generated with document and section info
    assert c.context_prefix is not None
    assert "Soft_Skills_III_Unit_I_MS_Excel_Basics.docx" in c.context_prefix
    assert "Statistical Functions" in c.context_prefix
    assert "Page: 3" in c.context_prefix
    # Check that full text has prefix prepended
    assert c.text.startswith(c.context_prefix)
    assert c.raw_text in c.text
    # Check that parent text also receives the prefix
    assert c.parent_text is not None
    assert c.parent_text.startswith(c.context_prefix)
    # Check serialization
    c_dict = c.to_dict()
    assert "context_prefix" in c_dict
    assert c_dict["context_prefix"] == c.context_prefix
    assert c_dict["raw_text"] == c.raw_text


def test_parent_child_enrichment():
    chunker = DocumentChunker(
        strategy="structure_aware",
        chunk_size=80,
        parent_chunk_size=180,
        enable_parent_child=True
    )
    sections = [
        DocumentSection(
            text="Block 1: Vector storage with ChromaDB.\nBlock 2: Lexical matching with BM25.\nBlock 3: Cross-encoder reranking.",
            section_title="Pipeline"
        )
    ]
    chunks = chunker.chunk_document(
        document_id="doc-pc",
        filename="pipeline.txt",
        file_type="txt",
        document_hash="hash-pc",
        sections=sections
    )

    assert len(chunks) >= 1
    # Check that parent_chunk_id and parent_text are populated
    for chunk in chunks:
        assert chunk.parent_chunk_id is not None
        assert chunk.parent_text is not None
        assert len(chunk.parent_text) >= len(chunk.text)


def test_parent_child_respects_page_boundaries():
    chunker = DocumentChunker(
        strategy="structure_aware",
        chunk_size=100,
        parent_chunk_size=5000,
        enable_parent_child=True
    )
    sections = [
        DocumentSection(text="Page 1 sentence content here.", page_number=1),
        DocumentSection(text="Page 2 sentence content here.", page_number=2),
    ]
    chunks = chunker.chunk_document(
        document_id="doc-p12",
        filename="multipage.pdf",
        file_type="pdf",
        document_hash="hash-p12",
        sections=sections
    )

    assert len(chunks) == 2
    # Ensure chunk 0 parent text does not contain Page 2 content
    assert "Page 2" not in chunks[0].parent_text
    # Ensure chunk 1 parent text does not contain Page 1 content
    assert "Page 1" not in chunks[1].parent_text
    # Parent chunk IDs must be different across pages
    assert chunks[0].parent_chunk_id != chunks[1].parent_chunk_id
