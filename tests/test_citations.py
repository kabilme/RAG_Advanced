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


def test_extract_citations_regex_flexibility():
    text = (
        "Item 1 [Source: report.pdf, Page 4] "
        "Item 2 [Source: slides.pptx, Slide: 12] "
        "Item 3 [Source: data.xlsx; Sheet: Revenue] "
        "Item 4 [Source: text.pdf, p. 7] "
        "Item 5 [Source: general.txt]"
    )
    citations = CitationManager.extract_citations(text)
    assert len(citations) == 5
    assert citations[0][1] == "report.pdf"
    assert citations[0][2] == "Page"
    assert citations[0][3] == "4"

    assert citations[1][1] == "slides.pptx"
    assert citations[1][2] == "Slide"
    assert citations[1][3] == "12"

    assert citations[2][1] == "data.xlsx"
    assert citations[2][2] == "Sheet"
    assert citations[2][3] == "Revenue"

    assert citations[3][1] == "text.pdf"
    assert citations[3][2] == "Page"
    assert citations[3][3] == "7"

    assert citations[4][1] == "general.txt"
    assert citations[4][2] is None


def test_validate_citations_autocorrects_hallucinated_page():
    chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="doc-1",
        text="Ratna started a food stall and rent doubled unexpectedly.",
        score=0.92,
        rank=1,
        retrieval_method="hybrid",
        filename="hees107.pdf",
        page_number=21,
    )
    # LLM cited printed book page 164 instead of PDF physical page 21
    text = "Ratna faced issues when rent doubled [Source: hees107.pdf, Page: 164]."
    items, sanitized = CitationManager.validate_citations(text, [chunk])

    assert len(items) == 1
    assert items[0].is_valid is True
    assert items[0].location_value == "21"
    assert items[0].filename == "hees107.pdf"
    # The citation in the text should be corrected to Page 21
    assert "[Source: hees107.pdf, Page: 21]" in sanitized
    assert "Page: 164" not in sanitized


def test_validate_citations_strips_ungrounded_fabricated_location():
    chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="doc-1",
        text="Solar energy efficiency is approximately 22 percent.",
        score=0.92,
        rank=1,
        retrieval_method="hybrid",
        filename="solar.pdf",
        page_number=4,
    )
    # The claim is about nuclear energy which has zero overlap with the solar chunk
    text = "Nuclear fusion reactors were installed across Europe [Source: solar.pdf, Page: 99]."
    items, sanitized = CitationManager.validate_citations(text, [chunk])

    assert len(items) == 1
    assert items[0].is_valid is False
    assert "[Source: solar.pdf, Page: 99]" not in sanitized


def test_validate_citations_blocks_cross_document_mismatch():
    # Demonstrates blocking citations where doc/page exists but statement has zero textual overlap
    chunk_excel = RetrievedChunk(
        chunk_id="c_excel",
        document_id="doc-excel",
        text="Excel text functions include LEFT, RIGHT, MID, CONCATENATE, and LEN.",
        score=0.45,
        rank=2,
        retrieval_method="hybrid",
        filename="Soft_Skills_Excel.docx",
        section="Text Functions",
    )
    chunk_ecosystem = RetrievedChunk(
        chunk_id="c_eco",
        document_id="doc-eco",
        text="Ecosystem biodiversity in oceans and rainforests stabilizes global climate.",
        score=0.92,
        rank=1,
        retrieval_method="hybrid",
        filename="hees101.pdf",
        page_number=6,
    )

    # LLM mistakenly cited Excel docx for an ecosystem claim
    answer = (
        "Ecosystem biodiversity in oceans and rainforests stabilizes climate "
        "[Source: Soft_Skills_Excel.docx, Section: Text Functions]."
    )

    items, sanitized = CitationManager.validate_citations(answer, [chunk_ecosystem, chunk_excel])

    # CitationManager detects zero overlap with the Excel chunk and auto-corrects to the matching ecosystem chunk
    assert len(items) == 1
    assert items[0].is_valid is True
    assert items[0].filename == "hees101.pdf"
    assert items[0].location_value == "6"
    assert "[Source: hees101.pdf, Page: 6]" in sanitized
    assert "Soft_Skills_Excel.docx" not in sanitized


def test_validate_citations_preserves_multiline_structure_and_paragraphs():
    multiline_text = (
        "### Section Title\n\n"
        "Introductory paragraph with context.\n\n"
        "Similarities:\n\n"
        "- Item One: Description here.\n"
        "- Item Two: Another description here.\n\n"
        "Differences:\n\n"
        "- Item Three: Distinct description here."
    )
    items, sanitized = CitationManager.validate_citations(multiline_text, [])
    assert "\n\n" in sanitized
    assert "Similarities:\n\n- Item One" in sanitized
    assert "Differences:\n\n- Item Three" in sanitized
    # Confirm it does not squash into a single line
    assert len(sanitized.splitlines()) >= 8
