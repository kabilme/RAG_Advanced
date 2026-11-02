"""Unit tests for citation extraction, validation, and hallucination prevention."""

import pytest
from app.retrieval.base import RetrievedChunk
from app.generation.citation_manager import CitationManager


@pytest.fixture
def retrieved_chunks():
    return [
        RetrievedChunk(
            chunk_id="c1",
            document_id="doc-1",
            text="Solar panels convert photons into electricity with 22% average efficiency.",
            score=0.95,
            rank=1,
            retrieval_method="hybrid",
            filename="solar_report.pdf",
            page_number=4,
        ),
        RetrievedChunk(
            chunk_id="c2",
            document_id="doc-2",
            text="Wind power capacity reached 906 gigawatts worldwide.",
            score=0.88,
            rank=2,
            retrieval_method="hybrid",
            filename="wind_overview.docx",
            section="Global Stats",
        ),
    ]


def test_extract_citations():
    text = (
        "Solar efficiency averages 22% [Source: solar_report.pdf, Page: 4]. "
        "Global capacity is expanding [Source: wind_overview.docx, Section: Global Stats]."
    )
    citations = CitationManager.extract_citations(text)
    assert len(citations) == 2
    assert citations[0][1] == "solar_report.pdf"
    assert citations[0][2] == "Page"
    assert citations[0][3] == "4"

    assert citations[1][1] == "wind_overview.docx"
    assert citations[1][2] == "Section"
    assert citations[1][3] == "Global Stats"


def test_validate_citations_with_valid_and_hallucinated(retrieved_chunks):
    # Text contains 1 valid citation and 1 fabricated citation for an unretrieved file
    text = (
        "Solar efficiency is 22% [Source: solar_report.pdf, Page: 4]. "
        "Geothermal is also growing [Source: fake_geothermal.pdf, Page: 10]."
    )

    items, sanitized_text = CitationManager.validate_citations(text, retrieved_chunks)

    assert len(items) == 2
    valid_items = [i for i in items if i.is_valid]
    invalid_items = [i for i in items if not i.is_valid]

    assert len(valid_items) == 1
    assert valid_items[0].filename == "solar_report.pdf"
    assert valid_items[0].location_value == "4"

    assert len(invalid_items) == 1
    assert invalid_items[0].filename == "fake_geothermal.pdf"

    # Fabricated citation should be stripped from output text
    assert "[Source: fake_geothermal.pdf, Page: 10]" not in sanitized_text
    assert "[Source: solar_report.pdf, Page: 4]" in sanitized_text


def test_source_cards_builder(retrieved_chunks):
    cards = CitationManager.build_source_cards(retrieved_chunks)
    assert len(cards) == 2
    assert cards[0]["filename"] == "solar_report.pdf"
    assert cards[0]["page_number"] == 4
    assert cards[0]["similarity_score"] == 0.95
