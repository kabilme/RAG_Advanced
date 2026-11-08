"""Database repositories for document tracking and conversation memory."""

import json
from typing import List, Optional
import uuid
from sqlalchemy.orm import Session
from app.database.models import DocumentRecord, ConversationRecord, MessageRecord


class DocumentRepository:
    """Manages document metadata persistence in SQLite."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, document_id: str) -> Optional[DocumentRecord]:
        return self.session.query(DocumentRecord).filter(DocumentRecord.document_id == document_id).first()

    def get_by_hash(self, file_hash: str) -> Optional[DocumentRecord]:
        return self.session.query(DocumentRecord).filter(DocumentRecord.file_hash == file_hash).first()

    def get_by_filename(self, filename: str) -> Optional[DocumentRecord]:
        return self.session.query(DocumentRecord).filter(DocumentRecord.filename == filename).first()

    def list_all(self) -> List[DocumentRecord]:
        return self.session.query(DocumentRecord).order_by(DocumentRecord.created_at.desc()).all()

    def upsert(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        file_hash: str,
        file_size_bytes: int,
        chunk_count: int,
        status: str = "indexed",
    ) -> DocumentRecord:
        record = self.get_by_hash(file_hash) or self.get_by_id(document_id)
        if not record:
            record = DocumentRecord(
                document_id=document_id,
                filename=filename,
                file_type=file_type,
                file_hash=file_hash,
                file_size_bytes=file_size_bytes,
                chunk_count=chunk_count,
                status=status,
            )
            self.session.add(record)
        else:
            record.filename = filename
            record.file_type = file_type
            record.file_hash = file_hash
            record.file_size_bytes = file_size_bytes
            record.chunk_count = chunk_count
            record.status = status

        try:
            self.session.commit()
            self.session.refresh(record)
            return record
        except Exception:
            self.session.rollback()
            raise

    def delete(self, document_id: str) -> bool:
        record = self.get_by_id(document_id)
        if record:
            try:
                self.session.delete(record)
                self.session.commit()
                return True
            except Exception:
                self.session.rollback()
                raise
        return False


class ConversationRepository:
    """Manages conversations and message turns in SQLite."""

    def __init__(self, session: Session):
        self.session = session

    def get_or_create_session(self, session_id: Optional[str] = None, title: Optional[str] = None) -> ConversationRecord:
        sid = session_id or str(uuid.uuid4())
        conv = self.session.query(ConversationRecord).filter(ConversationRecord.session_id == sid).first()
        if not conv:
            conv = ConversationRecord(session_id=sid, title=title or "New Conversation")
            self.session.add(conv)
            try:
                self.session.commit()
                self.session.refresh(conv)
            except Exception:
                self.session.rollback()
                raise
        return conv

    def list_sessions(self) -> List[ConversationRecord]:
        return self.session.query(ConversationRecord).order_by(ConversationRecord.updated_at.desc()).all()

    def get_messages(self, session_id: str, limit: int = 50) -> List[MessageRecord]:
        return (
            self.session.query(MessageRecord)
            .filter(MessageRecord.session_id == session_id)
            .order_by(MessageRecord.created_at.asc())
            .limit(limit)
            .all()
        )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[dict]] = None,
    ) -> MessageRecord:
        self.get_or_create_session(session_id)
        msg = MessageRecord(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            sources_json=json.dumps(sources) if sources else None,
        )
        self.session.add(msg)
        try:
            self.session.commit()
            self.session.refresh(msg)
            return msg
        except Exception:
            self.session.rollback()
            raise

    def delete_session(self, session_id: str) -> bool:
        conv = self.session.query(ConversationRecord).filter(ConversationRecord.session_id == session_id).first()
        if conv:
            try:
                self.session.delete(conv)
                self.session.commit()
                return True
            except Exception:
                self.session.rollback()
                raise
        return False
