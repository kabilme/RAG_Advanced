"""Unit tests for SQLite relational models and repositories."""

from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base
from app.database.repository import DocumentRepository, ConversationRepository


@pytest.fixture
def in_memory_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


def test_document_repository(in_memory_session):
    repo = DocumentRepository(in_memory_session)

    # Upsert document
    doc = repo.upsert(
        document_id="doc-001",
        filename="report.pdf",
        file_type="pdf",
        file_hash="sha256-hash-value-1",
        file_size_bytes=1024,
        chunk_count=5,
    )
    assert doc.document_id == "doc-001"
    assert doc.chunk_count == 5

    # Get by ID and hash
    by_id = repo.get_by_id("doc-001")
    assert by_id is not None
    assert by_id.filename == "report.pdf"

    by_hash = repo.get_by_hash("sha256-hash-value-1")
    assert by_hash is not None
    assert by_hash.document_id == "doc-001"

    # List all
    all_docs = repo.list_all()
    assert len(all_docs) == 1

    # Delete
    deleted = repo.delete("doc-001")
    assert deleted is True
    assert repo.get_by_id("doc-001") is None


def test_conversation_repository(in_memory_session):
    repo = ConversationRepository(in_memory_session)

    # Create session
    session = repo.get_or_create_session("session-abc", title="RAG Discussion")
    assert session.session_id == "session-abc"
    assert session.title == "RAG Discussion"

    # Add messages
    m1 = repo.add_message("session-abc", role="user", content="How does hybrid retrieval work?")
    m2 = repo.add_message(
        "session-abc",
        role="assistant",
        content="Hybrid retrieval combines BM25 and vector search using RRF.",
        sources=[{"filename": "report.pdf", "page": 2}]
    )

    assert m1.content == "How does hybrid retrieval work?"
    assert m2.sources[0]["filename"] == "report.pdf"

    messages = repo.get_messages("session-abc")
    assert len(messages) == 2

    # Delete session
    assert repo.delete_session("session-abc") is True
    assert len(repo.get_messages("session-abc")) == 0
