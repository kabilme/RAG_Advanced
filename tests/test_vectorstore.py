"""Unit and integration tests for ChromaDB vector store."""

from pathlib import Path
import pytest
from app.ingestion.chunking import TextChunk
from app.vectorstore.chroma_store import ChromaVectorStore


@pytest.fixture
def temp_chroma_dir(tmp_path: Path) -> Path:
    chroma_dir = tmp_path / "chroma_test"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir


def test_chroma_store_crud_and_search(temp_chroma_dir: Path):
    store = ChromaVectorStore(
        persist_directory=str(temp_chroma_dir),
        embedding_model_name="test_model",
        embedding_dimension=4,
    )

    chunks = [
        TextChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="Antigravity local RAG system running offline.",
            chunk_index=0,
            filename="rag.pdf",
            file_type="pdf",
            document_hash="hash-1",
            page_number=1,
            section="Intro",
        ),
        TextChunk(
            chunk_id="chunk-2",
            document_id="doc-2",
            text="ChromaDB handles vector storage and persistent embeddings.",
            chunk_index=0,
            filename="db.docx",
            file_type="docx",
            document_hash="hash-2",
            section="Database",
        ),
    ]
    # Unit vectors for test dimension 4
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    store.add_chunks(chunks, embeddings)
    assert store.count() == 2

    # Query matching chunk 1
    results = store.similarity_search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].similarity_score > 0.9

    # Query with metadata filter
    filtered = store.similarity_search(
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        top_k=2,
        filter_criteria={"document_id": "doc-2"}
    )
    assert len(filtered) == 1
    assert filtered[0].chunk_id == "chunk-2"

    # Test document deletion
    deleted_count = store.delete_document("doc-1")
    assert deleted_count == 1
    assert store.count() == 1

    # Verify doc-1 is gone
    doc1_chunks = store.get_document_chunks("doc-1")
    assert len(doc1_chunks) == 0


def test_chroma_store_backup_and_restore(temp_chroma_dir: Path, tmp_path: Path):
    store = ChromaVectorStore(
        persist_directory=str(temp_chroma_dir),
        embedding_model_name="backup_test",
        embedding_dimension=3,
    )
    chunks = [
        TextChunk(
            chunk_id="bk-1",
            document_id="doc-bk",
            text="Backup test chunk content.",
            chunk_index=0,
            filename="bk.txt",
            file_type="txt",
            document_hash="hash-bk",
        )
    ]
    embeddings = [[0.577, 0.577, 0.577]]
    store.add_chunks(chunks, embeddings)
    assert store.count() == 1

    backup_file = tmp_path / "chroma_backup.json"
    exported = store.backup_to_json(backup_file)
    assert exported == 1
    assert backup_file.exists()

    # Clear collection
    store.clear()
    assert store.count() == 0

    # Restore from backup
    restored = store.restore_from_json(backup_file)
    assert restored == 1
    assert store.count() == 1
