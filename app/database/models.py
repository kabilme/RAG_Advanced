"""SQLAlchemy ORM models for document management and conversation memory."""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    Text,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocumentRecord(Base):
    """Tracks uploaded and indexed document metadata."""
    __tablename__ = "documents"

    document_id = Column(String(64), primary_key=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(32), nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)
    file_size_bytes = Column(Integer, nullable=False, default=0)
    chunk_count = Column(Integer, nullable=False, default=0)
    status = Column(String(32), nullable=False, default="indexed")
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_hash": self.file_hash,
            "file_size_bytes": self.file_size_bytes,
            "chunk_count": self.chunk_count,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ConversationRecord(Base):
    """Tracks user chat sessions."""
    __tablename__ = "conversations"

    session_id = Column(String(64), primary_key=True)
    title = Column(String(255), nullable=False, default="New Conversation")
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    messages = relationship(
        "MessageRecord",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="MessageRecord.created_at"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": len(self.messages) if self.messages else 0,
        }


class MessageRecord(Base):
    """Stores individual conversational turns."""
    __tablename__ = "messages"

    message_id = Column(String(64), primary_key=True)
    session_id = Column(String(64), ForeignKey("conversations.session_id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    sources_json = Column(Text, nullable=True)  # JSON-encoded citation sources
    created_at = Column(DateTime, default=utcnow)

    conversation = relationship("ConversationRecord", back_populates="messages")

    @property
    def sources(self) -> List[Dict[str, Any]]:
        if not self.sources_json:
            return []
        try:
            return json.loads(self.sources_json)
        except Exception:
            return []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "sources": self.sources,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
