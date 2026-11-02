"""API route blueprints."""

from app.api.routes.health import router as health_router
from app.api.routes.documents import router as documents_router
from app.api.routes.chat import router as chat_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.settings import router as settings_router

__all__ = [
    "health_router",
    "documents_router",
    "chat_router",
    "retrieval_router",
    "settings_router",
]
