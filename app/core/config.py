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
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 1536
    NUM_THREADS: int = 1
    NUM_CTX: int = 6144
    TIMEOUT_SECONDS: int = 300

    # Hardware & Acceleration Settings (manual activation only)
    ENABLE_ACCELERATION: bool = False
    ACCELERATION_DEVICE: Literal["cpu", "gpu", "auto"] = "cpu"
    ACCELERATION_THREADS: int = 4

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
    ENABLE_CONTEXTUAL_RETRIEVAL: bool = True
    CONTEXTUAL_MODE: Literal["structural", "llm"] = "structural"

    # Retrieval & Ranking Settings
    RETRIEVAL_STRATEGY: Literal["hybrid", "semantic", "keyword"] = "hybrid"
    TOP_K: int = 15
    SIMILARITY_THRESHOLD: float = 0.3
    RRF_K: int = 60
    ENABLE_RERANKER: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K: int = 5
    CONTEXT_COMPRESSION: bool = True

    # Security & Processing
    MAX_QUERY_LENGTH: int = 1000
    ENABLE_OCR: bool = False
    ENABLE_VISION: bool = False
    VISION_MODEL: str = "moondream"
    VISION_MIN_IMAGE_SIZE: int = 4096
    VISION_MIN_DIMENSION: int = 100
    VISION_MAX_IMAGES_PER_PAGE: int = 3
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

    def is_gpu_available(self) -> bool:
        """Checks if a CUDA-compatible GPU is available."""
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def get_compute_device(self) -> str:
        """Determines the active compute device for local neural models."""
        if not self.ENABLE_ACCELERATION:
            return "cpu"

        dev = self.ACCELERATION_DEVICE.lower()
        if dev in ("gpu", "cuda"):
            if self.is_gpu_available():
                return "cuda"
            return "cpu"
        elif dev == "auto":
            return "cuda" if self.is_gpu_available() else "cpu"
        return "cpu"

    def get_num_threads(self) -> int:
        """Returns the number of threads to utilize based on acceleration state."""
        if self.ENABLE_ACCELERATION:
            return max(1, self.ACCELERATION_THREADS)
        return 1

    def apply_acceleration_settings(self) -> None:
        """Applies thread limits and acceleration settings to runtime libraries."""
        threads = self.get_num_threads()
        try:
            import torch
            torch.set_num_threads(threads)
        except Exception:
            pass
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)

    def update_settings(self, updates: dict) -> None:
        """Updates in-memory settings and persists them to the .env file."""
        for k, v in updates.items():
            if v is not None:
                attr = k.upper()
                if hasattr(self, attr):
                    cur_val = getattr(self, attr)
                    if isinstance(cur_val, bool) and isinstance(v, str):
                        casted = v.lower() in ("true", "1", "yes")
                    elif isinstance(cur_val, int) and not isinstance(v, bool):
                        casted = int(v)
                    elif isinstance(cur_val, float):
                        casted = float(v)
                    else:
                        casted = v
                    setattr(self, attr, casted)

        # Synchronize NUM_THREADS with acceleration state
        self.NUM_THREADS = self.get_num_threads()
        self.apply_acceleration_settings()

        env_path = Path(".env")
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

        updated_keys = set()
        new_lines = []
        for line in lines:
            trimmed = line.strip()
            if trimmed and not trimmed.startswith("#") and "=" in trimmed:
                key, _ = trimmed.split("=", 1)
                key_clean = key.strip()
                match_val = None
                for u_k, u_v in updates.items():
                    if u_v is not None and u_k.upper() == key_clean:
                        match_val = str(u_v)
                        break
                if match_val is not None:
                    new_lines.append(f"{key_clean}={match_val}")
                    updated_keys.add(key_clean)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        for u_k, u_v in updates.items():
            if u_v is not None:
                u_key = u_k.upper()
                if u_key not in updated_keys and hasattr(self, u_key):
                    new_lines.append(f"{u_key}={u_v}")

        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


# Global singleton settings instance
settings = Settings()
settings.ensure_directories()
settings.apply_acceleration_settings()
