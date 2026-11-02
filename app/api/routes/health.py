"""Health and system status routes."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import settings
from app.api.schemas import HealthResponse
from app.api.dependencies import (
    get_ollama_client,
    get_embedding_service,
    get_vector_store,
    get_db_session,
)
from app.database.repository import DocumentRepository

router = APIRouter(prefix="", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_system_health(
    ollama_client=Depends(get_ollama_client),
    embedding_service=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
    db: Session = Depends(get_db_session),
):
    """Returns comprehensive system health, Ollama status, and storage metrics."""
    ollama_health = ollama_client.check_health()
    doc_repo = DocumentRepository(db)
    docs = doc_repo.list_all()

    return HealthResponse(
        status="healthy",
        ollama_connected=ollama_health["connected"],
        configured_model=ollama_client.model_name,
        model_available=ollama_health["model_available"],
        embedding_model=embedding_service.model_name,
        embedding_dimension=embedding_service.dimension,
        vector_store_type="ChromaDB",
        indexed_vectors=vector_store.count(),
        total_documents=len(docs),
        installed_models=ollama_health.get("installed_models", []),
    )
