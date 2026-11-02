"""Chat and conversational question-answering endpoints."""

import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    CitationResponse,
    SourceCardResponse,
    ChatMessageResponse,
    ConversationSessionResponse,
)
from app.api.dependencies import get_rag_chain
from app.memory.conversation_manager import ConversationManager

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/query", response_model=QueryResponse)
def submit_chat_query(
    request: QueryRequest,
    rag_chain=Depends(get_rag_chain),
):
    """Answers user question using grounded local RAG pipeline with verified citations."""
    try:
        response = rag_chain.query(
            question=request.question,
            session_id=request.session_id,
            filter_criteria=request.filter_criteria,
            top_k=request.top_k,
        )

        citation_objs = [CitationResponse(**c) for c in response.citations]
        source_card_objs = [SourceCardResponse(**sc) for sc in response.source_cards]

        return QueryResponse(
            question=response.question,
            answer=response.answer,
            session_id=response.session_id or "",
            is_grounded=response.is_grounded,
            is_abstained=response.is_abstained,
            citations=citation_objs,
            source_cards=source_card_objs,
            warnings=response.validation_warnings,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post("/stream")
def stream_chat_query(
    request: QueryRequest,
    rag_chain=Depends(get_rag_chain),
):
    """Streams RAG generation tokens and source citations in SSE format."""
    def event_generator():
        try:
            for event in rag_chain.stream_query(
                question=request.question,
                session_id=request.session_id,
                filter_criteria=request.filter_criteria,
                top_k=request.top_k,
            ):
                payload = json.dumps(event)
                yield f"data: {payload}\n\n"
        except Exception as exc:
            err_payload = json.dumps({"type": "error", "message": str(exc)})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/sessions", response_model=List[ConversationSessionResponse])
def list_chat_sessions():
    """Lists all conversation sessions."""
    mem = ConversationManager()
    sessions = mem.list_sessions()
    mem.close()
    return [ConversationSessionResponse(**s) for s in sessions]


@router.get("/history/{session_id}", response_model=List[ChatMessageResponse])
def get_chat_history(session_id: str):
    """Retrieves full message turn history for a conversation session."""
    mem = ConversationManager()
    messages = mem.get_full_session_messages(session_id)
    mem.close()
    return [ChatMessageResponse(**m) for m in messages]


@router.delete("/history/{session_id}")
def delete_chat_session(session_id: str):
    """Deletes a conversation session."""
    mem = ConversationManager()
    success = mem.delete_session(session_id)
    mem.close()
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found.")
    return {"message": f"Successfully deleted conversation session {session_id}"}
