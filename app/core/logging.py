"""Centralized logging configuration for the Local Advanced RAG system."""

import logging
import sys
from app.core.config import settings

_log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logger(name: str = "rag_advanced") -> logging.Logger:
    """Configures and returns a logger instance with consistent formatting."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        level = _log_levels.get(settings.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()
