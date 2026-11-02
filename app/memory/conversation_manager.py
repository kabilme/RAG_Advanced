"""Conversation memory management with persistence and sliding window summarization."""

from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.database.database import SessionLocal
from app.database.repository import ConversationRepository


class ConversationManager:
    """Manages multi-turn conversation sessions, persistence, and history windows."""

    def __init__(self, db_session: Optional[Session] = None, max_history_turns: int = 10):
        self._session_provided = db_session is not None
        self._db = db_session or SessionLocal()
        self._repo = ConversationRepository(self._db)
        self.max_history_turns = max_history_turns

    def close(self) -> None:
        """Closes internal database session if locally created."""
        if not self._session_provided and self._db:
            self._db.close()

    def get_or_create_session(self, session_id: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
        """Fetches or creates a chat session."""
        conv = self._repo.get_or_create_session(session_id=session_id, title=title)
        return conv.to_dict()

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists all existing chat sessions."""
        convs = self._repo.list_sessions()
        return [c.to_dict() for c in convs]

    def add_user_message(self, session_id: str, content: str) -> Dict[str, Any]:
        """Records a user question."""
        msg = self._repo.add_message(session_id=session_id, role="user", content=content)
        return msg.to_dict()

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Records the assistant answer along with verified citation sources."""
        msg = self._repo.add_message(
            session_id=session_id,
            role="assistant",
            content=content,
            sources=sources,
        )
        return msg.to_dict()

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Retrieves conversational history formatted as a list of role/content dicts."""
        max_turns = limit or self.max_history_turns
        records = self._repo.get_messages(session_id=session_id, limit=max_turns * 2)
        history = []
        for r in records:
            history.append({
                "role": r.role,
                "content": r.content,
            })
        return history

    def get_full_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves complete message records including citations."""
        records = self._repo.get_messages(session_id=session_id, limit=100)
        return [r.to_dict() for r in records]

    def delete_session(self, session_id: str) -> bool:
        """Deletes a chat session and all associated turns."""
        success = self._repo.delete_session(session_id)
        logger.info("Deleted conversation session: %s", session_id)
        return success
