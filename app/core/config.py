"""Application configuration management using Pydantic Settings."""

import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class Settings(BaseSettings):
    """Central configuration for Local Advanced RAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server settings
    SERVER_HOST: str = "127.0.0.1"
    SERVER_PORT: int = 8000
    RELOAD: bool = True

    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 2048
    TIMEOUT_SECONDS: int = 120

    # Storage Paths
    VECTOR_DB_PATH: str = "./data/chroma_db"
    UPLOAD_DIR: str = "./data/uploads"
    PROCESSED_DIR: str = "./data/processed"
    EVALUATION_DIR: str = "./data/evaluations"
    SQLITE_DB_PATH: str = "./data/rag_app.db"
    MAX_UPLOAD_SIZE_MB: int = 50

    # Chunking Configuration
    CHUNK_STRATEGY: Literal[
        "structure_aware",
        "recursive_character",
        "sentence",
        "paragraph",
        "fixed_size"
    ] = "structure_aware"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    PARENT_CHUNK_SIZE: int = 2500
    ENABLE_PARENT_CHILD: bool = True

    # Retrieval & Ranking Settings
    RETRIEVAL_STRATEGY: Literal["hybrid", "semantic", "keyword"] = "hybrid"
    TOP_K: int = 15
    SIMILARITY_THRESHOLD: float = 0.3
    RRF_K: int = 60
    ENABLE_RERANKER: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K: int = 8
    CONTEXT_COMPRESSION: bool = True

    # Security & Processing
    MAX_QUERY_LENGTH: int = 1000
    ENABLE_OCR: bool = False
    LOG_LEVEL: str = "INFO"

    def ensure_directories(self) -> None:
        """Create necessary directories on disk if they do not exist."""
        for path_attr in [
            self.VECTOR_DB_PATH,
            self.UPLOAD_DIR,
            self.PROCESSED_DIR,
            self.EVALUATION_DIR,
            os.path.dirname(self.SQLITE_DB_PATH)
        ]:
            if path_attr:
                Path(path_attr).mkdir(parents=True, exist_ok=True)


# Global singleton settings instance
settings = Settings()
settings.ensure_directories()
