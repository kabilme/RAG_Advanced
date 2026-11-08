"""Settings and configuration management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.api.schemas import SettingsUpdateRequest, SettingsResponse
from app.api.dependencies import get_ollama_client
from app.ingestion.vision_processor import VisionProcessor

router = APIRouter(prefix="/settings", tags=["Settings"])


@router.get("", response_model=SettingsResponse)
def get_current_settings(ollama_client=Depends(get_ollama_client)):
    """Returns the current system configuration and available Ollama models."""
    ollama_health = ollama_client.check_health()
    installed = ollama_health.get("installed_models", [])

    return SettingsResponse(
        llm_model=settings.LLM_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        chunk_strategy=settings.CHUNK_STRATEGY,
        retrieval_strategy=settings.RETRIEVAL_STRATEGY,
        top_k=settings.TOP_K,
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
        enable_reranker=settings.ENABLE_RERANKER,
        reranker_model=settings.RERANKER_MODEL,
        rerank_top_k=settings.RERANK_TOP_K,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
        enable_vision=settings.ENABLE_VISION,
        vision_model=settings.VISION_MODEL,
        installed_models=installed,
    )


@router.post("", response_model=SettingsResponse)
def update_system_settings(
    update_req: SettingsUpdateRequest,
    ollama_client=Depends(get_ollama_client),
):
    """Updates runtime configuration and persists changes to .env file."""
    try:
        updates = update_req.model_dump(exclude_unset=True)
        settings.update_settings(updates)
        VisionProcessor.reset_state()

        ollama_health = ollama_client.check_health()
        installed = ollama_health.get("installed_models", [])

        return SettingsResponse(
            llm_model=settings.LLM_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            chunk_strategy=settings.CHUNK_STRATEGY,
            retrieval_strategy=settings.RETRIEVAL_STRATEGY,
            top_k=settings.TOP_K,
            similarity_threshold=settings.SIMILARITY_THRESHOLD,
            enable_reranker=settings.ENABLE_RERANKER,
            reranker_model=settings.RERANKER_MODEL,
            rerank_top_k=settings.RERANK_TOP_K,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            enable_vision=settings.ENABLE_VISION,
            vision_model=settings.VISION_MODEL,
            installed_models=installed,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
