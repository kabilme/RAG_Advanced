"""Retrieval testing endpoints for standalone semantic, keyword, or hybrid search."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas import SearchRequest, SearchResponse, SourceCardResponse
from app.api.dependencies import get_retrieval_pipeline
from app.generation.citation_manager import CitationManager

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


@router.post("/search", response_model=SearchResponse)
def test_retrieval(
    request: SearchRequest,
    pipeline=Depends(get_retrieval_pipeline),
):
    """Executes retrieval without invoking LLM generation for testing search quality."""
    try:
        chunks = pipeline.retrieve(
            query=request.query,
            top_k=request.top_k,
            filter_criteria=request.filter_criteria,
        )

        cards = CitationManager.build_source_cards(chunks)
        card_objs = [SourceCardResponse(**c) for c in cards]

        return SearchResponse(
            query=request.query,
            results=card_objs,
            total_results=len(card_objs),
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
