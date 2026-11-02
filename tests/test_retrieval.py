"""Unit and integration tests for the retrieval pipeline."""

from unittest.mock import MagicMock
import pytest
from app.ingestion.chunking import TextChunk
from app.retrieval.base import RetrievedChunk
from app.retrieval.keyword_search import BM25SearchEngine
from app.retrieval.semantic_search import SemanticSearchEngine
from app.retrieval.hybrid_search import HybridSearchEngine
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.context_compressor import ContextCompressor
from app.retrieval.retrieval_pipeline import RetrievalPipeline


@pytest.fixture
def sample_chunks():
    return [
        TextChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="Solar energy systems produce clean electricity directly from sunlight.",
            chunk_index=0,
            filename="solar.pdf",
            file_type="pdf",
            document_hash="hash-solar",
            page_number=1,
            section="Overview",
        ),
        TextChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            text="The initial installation cost of photovoltaic panels remains relatively high.",
            chunk_index=1,
            filename="solar.pdf",
            file_type="pdf",
            document_hash="hash-solar",
            page_number=2,
            section="Costs",
        ),
        TextChunk(
            chunk_id="chunk-3",
            document_id="doc-2",
            text="Wind turbine efficiency depends on average local atmospheric wind speeds.",
            chunk_index=0,
            filename="wind.docx",
            file_type="docx",
            document_hash="hash-wind",
            section="Wind Energy",
        ),
    ]


def test_bm25_keyword_search(sample_chunks):
    bm25 = BM25SearchEngine()
    bm25.index_chunks(sample_chunks)

    # Search for specific lexical terms
    results = bm25.search("photovoltaic installation cost", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "chunk-2"
    assert "cost" in results[0].text.lower()

    # Search for non-existent keyword
    empty_results = bm25.search("geothermal magma", top_k=2)
    assert len(empty_results) == 0

    # Remove document
    removed = bm25.remove_document("doc-1")
    assert removed == 2
    post_remove = bm25.search("photovoltaic", top_k=2)
    assert len(post_remove) == 0


def test_query_rewriter():
    # Classification tests
    assert QueryRewriter.classify_query("Compare solar and wind energy") == "comparative"
    assert QueryRewriter.classify_query("Summarize the key findings of the report") == "summarization"
    assert QueryRewriter.classify_query("What is the cost of solar panels in 2024?") == "numerical"
    assert QueryRewriter.classify_query("Define photovoltaic effect") == "factual"

    # Coreference / follow-up resolution test
    history = [
        {"role": "user", "content": "What are the advantages of solar energy?"},
        {"role": "assistant", "content": "Solar energy produces zero greenhouse emissions."},
    ]
    resolved = QueryRewriter.resolve_followup("What are its disadvantages?", history)
    assert "solar energy" in resolved.lower()
    assert "disadvantages" in resolved.lower()

    # Query decomposition test
    multi_part = "What is solar energy and how does it compare to wind?"
    subqueries = QueryRewriter.decompose_query(multi_part)
    assert len(subqueries) >= 2


def test_hybrid_search_rrf(sample_chunks):
    mock_vector_store = MagicMock()
    mock_embedding_service = MagicMock()
    mock_embedding_service.embed_query.return_value = [0.1, 0.2, 0.3]

    # Mock semantic search results
    semantic_engine = SemanticSearchEngine(mock_vector_store, mock_embedding_service)
    semantic_engine.search = MagicMock(return_value=[
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text=sample_chunks[0].text,
            score=0.92,
            rank=1,
            retrieval_method="semantic",
            filename="solar.pdf",
        ),
        RetrievedChunk(
            chunk_id="chunk-3",
            document_id="doc-2",
            text=sample_chunks[2].text,
            score=0.75,
            rank=2,
            retrieval_method="semantic",
            filename="wind.docx",
        ),
    ])

    keyword_engine = BM25SearchEngine()
    keyword_engine.index_chunks(sample_chunks)

    hybrid_engine = HybridSearchEngine(
        semantic_engine=semantic_engine,
        keyword_engine=keyword_engine,
        rrf_k=60,
        dense_weight=0.7,
        sparse_weight=0.3,
    )

    fused = hybrid_engine.search("solar electricity sunlight", top_k=2)
    assert len(fused) >= 1
    # chunk-1 appears in both dense and sparse results -> should win top rank!
    assert fused[0].chunk_id == "chunk-1"
    assert fused[0].retrieval_method == "hybrid"
    assert fused[0].score > 0.0


def test_context_compressor(sample_chunks):
    dup_chunk = RetrievedChunk(
        chunk_id="chunk-dup",
        document_id="doc-1",
        text="Solar energy systems produce clean electricity directly from sunlight.",
        score=0.9,
        rank=2,
        retrieval_method="semantic",
        filename="solar.pdf",
    )
    chunks = [
        RetrievedChunk(
            chunk_id=sample_chunks[0].chunk_id,
            document_id=sample_chunks[0].document_id,
            text=sample_chunks[0].text,
            score=0.95,
            rank=1,
            retrieval_method="semantic",
            filename=sample_chunks[0].filename,
        ),
        dup_chunk,
    ]

    deduped = ContextCompressor.deduplicate_chunks(chunks)
    assert len(deduped) == 1  # Near-duplicate eliminated


def test_retrieval_pipeline_end_to_end(sample_chunks):
    mock_semantic = MagicMock()
    mock_semantic.search.return_value = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text=sample_chunks[0].text,
            score=0.9,
            rank=1,
            retrieval_method="semantic",
            filename="solar.pdf",
        )
    ]

    mock_keyword = BM25SearchEngine()
    mock_keyword.index_chunks(sample_chunks)

    hybrid_engine = HybridSearchEngine(
        semantic_engine=mock_semantic,
        keyword_engine=mock_keyword,
    )

    reranker = CrossEncoderReranker(enabled=False)  # Test with reranker disabled fallback
    pipeline = RetrievalPipeline(hybrid_engine=hybrid_engine, reranker=reranker)

    history = [
        {"role": "user", "content": "Tell me about solar energy."},
        {"role": "assistant", "content": "Solar energy is renewable."},
    ]
    results = pipeline.retrieve("What are its primary benefits?", top_k=2, conversation_history=history)
    assert len(results) >= 1
    assert "Source: solar.pdf" in results[0].to_citation()
