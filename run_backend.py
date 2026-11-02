"""Backend launcher script."""

import uvicorn
from app.core.config import settings
from app.core.logging import logger

if __name__ == "__main__":
    logger.info("Starting Local Advanced RAG Backend on %s:%d", settings.SERVER_HOST, settings.SERVER_PORT)
    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
