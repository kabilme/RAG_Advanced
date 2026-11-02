"""FastAPI dependency injection providers."""

from functools import lru_cache
from typing import Generator
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.database import SessionLocal, init_db
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaVectorStore
from app.retrieval.semantic_search import SemanticSearchEngine
from app.retrieval.keyword_search import BM25SearchEngine
from app.retrieval.hybrid_search import HybridSearchEngine
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retrieval_pipeline import RetrievalPipeline
from app.generation.ollama_client import OllamaClient
from app.generation.rag_chain import RAGChain
from app.ingestion.document_loader import DocumentIngestionEngine
from app.memory.conversation_manager import ConversationManager


# Initialize DB tables on startup
init_db()

# Global singletons
_embedding_service: EmbeddingService = None
_vector_store: ChromaVectorStore = None
_keyword_engine: BM25SearchEngine = None
_reranker: CrossEncoderReranker = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(
            model_name=settings.EMBEDDING_MODEL,
            fallback_to_local=True
        )
    return _embedding_service


def get_vector_store() -> ChromaVectorStore:
    global _vector_store
    if _vector_store is None:
        emb_svc = get_embedding_service()
        _vector_store = ChromaVectorStore(
            persist_directory=settings.VECTOR_DB_PATH,
            embedding_model_name=emb_svc.model_name,
            embedding_dimension=emb_svc.dimension,
        )
    return _vector_store


def get_keyword_engine() -> BM25SearchEngine:
    global _keyword_engine
    if _keyword_engine is None:
        _keyword_engine = BM25SearchEngine()
        try:
            vs = get_vector_store()
            existing_chunks = vs.get_all_chunks()
            if existing_chunks:
                _keyword_engine.add_chunks(existing_chunks)
        except Exception:
            pass
    return _keyword_engine


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(
            model_name=settings.RERANKER_MODEL,
            enabled=settings.ENABLE_RERANKER
        )
    return _reranker


def get_retrieval_pipeline() -> RetrievalPipeline:
    vs = get_vector_store()
    emb = get_embedding_service()
    kw = get_keyword_engine()
    reranker = get_reranker()

    semantic_engine = SemanticSearchEngine(vector_store=vs, embedding_service=emb)
    hybrid_engine = HybridSearchEngine(
        semantic_engine=semantic_engine,
        keyword_engine=kw,
        rrf_k=settings.RRF_K,
    )

    return RetrievalPipeline(
        hybrid_engine=hybrid_engine,
        reranker=reranker,
        strategy=settings.RETRIEVAL_STRATEGY,
    )


def get_ollama_client() -> OllamaClient:
    return OllamaClient(
        base_url=settings.OLLAMA_BASE_URL,
        model_name=settings.LLM_MODEL,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
    )


def get_rag_chain() -> RAGChain:
    pipeline = get_retrieval_pipeline()
    ollama_c = get_ollama_client()
    mem = ConversationManager()
    return RAGChain(
        retrieval_pipeline=pipeline,
        ollama_client=ollama_c,
        conversation_manager=mem,
    )


def get_ingestion_engine() -> DocumentIngestionEngine:
    return DocumentIngestionEngine()


def get_db_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
