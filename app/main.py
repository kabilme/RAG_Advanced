"""FastAPI main application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import logger
from app.database.database import init_db
from app.api.routes import (
    health_router,
    documents_router,
    chat_router,
    retrieval_router,
    settings_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    logger.info("Initializing Local Advanced RAG Backend...")
    settings.ensure_directories()
    init_db()
    logger.info("Local Advanced RAG Backend initialized and ready on %s:%d", settings.SERVER_HOST, settings.SERVER_PORT)
    yield
    logger.info("Shutting down Local Advanced RAG Backend...")


app = FastAPI(
    title="Local Advanced RAG API",
    description="Privacy-preserving local RAG platform powered by Ollama, ChromaDB, and Sentence Transformers.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local Streamlit UI or web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(retrieval_router)
app.include_router(settings_router)


@app.get("/")
def root():
    return {
        "message": "Local Advanced RAG System API is running.",
        "docs_url": "/docs",
        "health_url": "/health",
    }
