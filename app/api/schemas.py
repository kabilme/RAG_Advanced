"""Pydantic request and response schemas for FastAPI endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    configured_model: str
    model_available: bool
    embedding_model: str
    embedding_dimension: int
    vector_store_type: str
    indexed_vectors: int
    total_documents: int
    installed_models: List[str] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_hash: str
    file_size_bytes: int
    chunk_count: int
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total_count: int


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    filter_criteria: Optional[Dict[str, Any]] = None


class CitationResponse(BaseModel):
    raw_citation: str
    filename: str
    location_type: Optional[str] = None
    location_value: Optional[str] = None
    is_valid: bool
    snippet: Optional[str] = None


class SourceCardResponse(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    filename: str
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    sheet_name: Optional[str] = None
    section: Optional[str] = None
    similarity_score: float
    citation: str
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    session_id: str
    is_grounded: bool
    is_abstained: bool
    citations: List[CitationResponse] = Field(default_factory=list)
    source_cards: List[SourceCardResponse] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    strategy: Optional[str] = "hybrid"  # semantic, keyword, hybrid
    filter_criteria: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SourceCardResponse]
    total_results: int


class ChatMessageResponse(BaseModel):
    message_id: str
    session_id: str
    role: str
    content: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: Optional[str] = None


class ConversationSessionResponse(BaseModel):
    session_id: str
    title: str
    message_count: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SettingsUpdateRequest(BaseModel):
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    enable_reranker: Optional[bool] = None
    rerank_top_k: Optional[int] = None
    temperature: Optional[float] = None


class SettingsResponse(BaseModel):
    llm_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    chunk_strategy: str
    retrieval_strategy: str
    top_k: int
    similarity_threshold: float
    enable_reranker: bool
    reranker_model: str
    rerank_top_k: int
    temperature: float
    max_tokens: int
    installed_models: List[str] = Field(default_factory=list)
