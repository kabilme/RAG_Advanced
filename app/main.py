"""FastAPI main application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import logger
from app.database.database import init_db
from app.api.routes import (
    health_router,
    documents_router,
    chat_router,
    retrieval_router,
    settings_router,
    widget_router,
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
app.include_router(widget_router)

# Mount static files directory for embeddable chatbot widget
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def root(request: Request):
    accept = request.headers.get("accept", "")
    index_file = static_dir / "index.html"
    if "text/html" in accept and "application/json" not in accept and index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "Local Advanced RAG System API is running.",
        "docs_url": "/docs",
        "health_url": "/health",
        "ui_url": "/ui",
        "widget_studio_url": "/widget/demo",
        "widget_embed_url": "/widget/embed",
    }


@app.get("/ui", include_in_schema=False)
@app.get("/app", include_in_schema=False)
def web_ui():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"error": "Web application index.html not found."}, status_code=404)
