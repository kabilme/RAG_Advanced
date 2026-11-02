"""Dedicated Ollama LLM client for streaming and synchronous inference."""

from typing import Any, Dict, Generator, List, Optional
import httpx
import ollama

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import (
    OllamaConnectionError,
    ModelNotAvailableError,
    RAGException,
)


class OllamaClient:
    """Encapsulates interaction with the locally hosted Ollama server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model_name = model_name or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.TEMPERATURE
        self.max_tokens = max_tokens or settings.MAX_TOKENS
        self.timeout = timeout or settings.TIMEOUT_SECONDS
        self._client = ollama.Client(host=self.base_url)

    def check_health(self) -> Dict[str, Any]:
        """Checks if Ollama server is running and queries available models.
        
        Returns:
            Dict containing 'connected' (bool), 'model_available' (bool), and 'models' (list).
        """
        try:
            resp = self._client.list()
            installed = []
            if hasattr(resp, "models"):
                for m in resp.models:
                    name = getattr(m, "model", None) or getattr(m, "name", "")
                    installed.append(name)
            elif isinstance(resp, dict) and "models" in resp:
                for m in resp["models"]:
                    installed.append(m.get("name", m.get("model", "")))

            # Check if model or model prefix is available
            target_prefix = self.model_name.split(":")[0]
            model_found = any(
                m == self.model_name or m.startswith(f"{target_prefix}:")
                for m in installed
            )

            return {
                "connected": True,
                "model_available": model_found,
                "configured_model": self.model_name,
                "installed_models": installed,
                "base_url": self.base_url,
            }
        except Exception as exc:
            logger.warning("Failed to connect to Ollama server at %s: %s", self.base_url, exc)
            return {
                "connected": False,
                "model_available": False,
                "configured_model": self.model_name,
                "installed_models": [],
                "base_url": self.base_url,
                "error": str(exc),
            }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generates a synchronous completion from the local Ollama LLM."""
        if not prompt.strip():
            raise RAGException("Prompt cannot be empty.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        options = self._build_options(temperature, max_tokens)

        try:
            logger.debug("Dispatching generation request to Ollama (model: %s)", self.model_name)
            response = self._client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
                stream=False,
            )
            # Extract content from response
            if hasattr(response, "message"):
                return response.message.content
            elif isinstance(response, dict) and "message" in response:
                return response["message"].get("content", "")
            return str(response)

        except Exception as exc:
            err_msg = str(exc).lower()
            if "connection" in err_msg or "connect" in err_msg:
                raise OllamaConnectionError(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?") from exc
            if "not found" in err_msg or "pull" in err_msg:
                raise ModelNotAvailableError(
                    f"Ollama model '{self.model_name}' is not installed. Run 'ollama pull {self.model_name}'."
                ) from exc
            raise RAGException(f"Ollama inference error: {exc}") from exc

    def _build_options(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Constructs CPU-optimized inference options for Ollama."""
        return {
            "temperature": temperature if temperature is not None else self.temperature,
            "num_predict": max_tokens or self.max_tokens,
            "num_thread": getattr(settings, "NUM_THREADS", 6),
            "num_ctx": getattr(settings, "NUM_CTX", 3072),
        }

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Streams generation tokens chunk-by-chunk from the local Ollama LLM."""
        if not prompt.strip():
            return

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        options = self._build_options(temperature, max_tokens)

        try:
            logger.debug("Starting streaming generation with Ollama (model: %s)", self.model_name)
            stream = self._client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
                stream=True,
            )
            for chunk in stream:
                token = ""
                if hasattr(chunk, "message"):
                    token = chunk.message.content or ""
                elif isinstance(chunk, dict) and "message" in chunk:
                    token = chunk["message"].get("content", "")
                if token:
                    yield token

        except Exception as exc:
            err_msg = str(exc).lower()
            if "connection" in err_msg or "connect" in err_msg:
                raise OllamaConnectionError(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?") from exc
            if "not found" in err_msg:
                raise ModelNotAvailableError(
                    f"Ollama model '{self.model_name}' is not installed. Run 'ollama pull {self.model_name}'."
                ) from exc
            raise RAGException(f"Ollama streaming inference error: {exc}") from exc
