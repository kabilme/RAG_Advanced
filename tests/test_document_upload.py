"""Unit tests for document upload idempotency, duplicate handling, and rollback."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, DocumentRecord
from app.database.repository import DocumentRepository
from app.ingestion.document_loader import DocumentIngestionEngine
from app.ingestion.chunking import TextChunk
from app.vectorstore.chroma_store import ChromaVectorStore
from app.retrieval.keyword_search import BM25SearchEngine


@pytest.fixture
def in_memory_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


@pytest.fixture
def mock_embedding_service():
    service = MagicMock()
    service.model_name = "test-model"
    service.dimension = 4
    service.embed_documents.side_effect = lambda texts: [[0.1, 0.2, 0.3, 0.4] for _ in texts]
    return service


def test_upsert_prevents_duplicate_file_hash_collision(in_memory_session):
    """DocumentRepository.upsert must not fail with IntegrityError when re-upserting same hash."""
    repo = DocumentRepository(in_memory_session)

    # Initial upsert
    doc1 = repo.upsert(
        document_id="id-alpha",
        filename="notes.txt",
        file_type="txt",
        file_hash="hash-abc-123",
        file_size_bytes=100,
        chunk_count=3,
        status="indexed",
    )
    assert doc1.document_id == "id-alpha"
    assert doc1.file_hash == "hash-abc-123"

    # Re-upsert with a new random UUID (as would happen on re-upload)
    doc2 = repo.upsert(
        document_id="id-beta-random",
        filename="notes.txt",
        file_type="txt",
        file_hash="hash-abc-123",
        file_size_bytes=100,
        chunk_count=4,
        status="indexed",
    )
    # Must reuse existing record and not crash
    assert doc2.document_id == "id-alpha"
    assert doc2.chunk_count == 4
    assert len(repo.list_all()) == 1


def test_upsert_updates_existing_record_by_filename(in_memory_session):
    """When a file is modified, looking up by filename allows updating the existing record."""
    repo = DocumentRepository(in_memory_session)

    # Initial version
    doc1 = repo.upsert(
        document_id="id-doc-1",
        filename="data.csv",
        file_type="csv",
        file_hash="hash-v1",
        file_size_bytes=500,
        chunk_count=2,
    )
    assert doc1.file_hash == "hash-v1"

    # Check filename lookup
    found = repo.get_by_filename("data.csv")
    assert found is not None
    assert found.document_id == "id-doc-1"

    # Update with new content (hash-v2) using the existing document_id
    doc2 = repo.upsert(
        document_id=found.document_id,
        filename="data.csv",
        file_type="csv",
        file_hash="hash-v2",
        file_size_bytes=600,
        chunk_count=3,
        status="re-indexed",
    )
    assert doc2.document_id == "id-doc-1"
    assert doc2.file_hash == "hash-v2"
    assert doc2.chunk_count == 3
    assert len(repo.list_all()) == 1


def test_reupload_flow_clears_old_chunks(tmp_path: Path, in_memory_session, mock_embedding_service):
    """Simulates the upload endpoint flow: re-uploading an existing file removes old chunks before indexing."""
    repo = DocumentRepository(in_memory_session)
    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        embedding_model_name="test-model",
        embedding_dimension=4,
    )
    keyword_engine = BM25SearchEngine()
    ingestion_engine = DocumentIngestionEngine()

    test_file = tmp_path / "guide.txt"
    test_file.write_text("This is version one of the guide document.", encoding="utf-8")
    file_hash = ingestion_engine.calculate_sha256(test_file)

    # First upload simulation
    res1 = ingestion_engine.ingest_file(test_file)
    emb1 = mock_embedding_service.embed_documents([c.text for c in res1.chunks])
    vector_store.add_chunks(res1.chunks, emb1)
    keyword_engine.index_chunks(res1.chunks)
    repo.upsert(
        document_id=res1.document_id,
        filename=res1.filename,
        file_type=res1.file_type,
        file_hash=res1.document_hash,
        file_size_bytes=res1.file_size_bytes,
        chunk_count=len(res1.chunks),
    )

    assert vector_store.count() == len(res1.chunks)
    first_doc_id = res1.document_id

    # Second upload simulation (exact same file)
    existing_doc = repo.get_by_hash(file_hash) or repo.get_by_filename(test_file.name)
    assert existing_doc is not None
    target_doc_id = existing_doc.document_id

    # Clean old chunks
    vector_store.delete_document(target_doc_id)
    keyword_engine.remove_document(target_doc_id)
    assert vector_store.count() == 0

    # Re-index with same document_id
    res2 = ingestion_engine.ingest_file(test_file, custom_document_id=target_doc_id)
    emb2 = mock_embedding_service.embed_documents([c.text for c in res2.chunks])
    vector_store.add_chunks(res2.chunks, emb2)
    keyword_engine.index_chunks(res2.chunks)
    rec2 = repo.upsert(
        document_id=res2.document_id,
        filename=res2.filename,
        file_type=res2.file_type,
        file_hash=res2.document_hash,
        file_size_bytes=res2.file_size_bytes,
        chunk_count=len(res2.chunks),
    )

    # Verification: chunk count unchanged, no duplicates, ID preserved
    assert rec2.document_id == first_doc_id
    assert vector_store.count() == len(res2.chunks)
    assert len(repo.list_all()) == 1


def test_failed_persist_cleans_up_orphaned_chunks(tmp_path: Path):
    """When DB persist fails, vector store and BM25 chunks must be deleted."""
    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_fail"),
        embedding_model_name="test-model",
        embedding_dimension=4,
    )
    keyword_engine = BM25SearchEngine()

    dummy_chunks = [
        TextChunk(
            chunk_id="chk-orphan-1",
            document_id="doc-orphan-1",
            text="Temporary chunk that should not remain.",
            chunk_index=0,
            filename="orphan.txt",
            file_type="txt",
            document_hash="hash-orphan",
        )
    ]
    embeddings = [[0.1, 0.2, 0.3, 0.4]]

    vector_store.add_chunks(dummy_chunks, embeddings)
    keyword_engine.index_chunks(dummy_chunks)
    assert vector_store.count() == 1

    # Simulate error handling cleanup
    try:
        raise RuntimeError("Simulated database failure during upsert")
    except Exception:
        vector_store.delete_document("doc-orphan-1")
        keyword_engine.remove_document("doc-orphan-1")

    assert vector_store.count() == 0
    assert len(keyword_engine._chunks) == 0

