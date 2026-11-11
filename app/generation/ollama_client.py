"""Dedicated Ollama LLM client for streaming and synchronous inference."""

from typing import Any, Dict, Generator, List, Optional
import queue
import threading
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
        self._client = ollama.Client(host=self.base_url, timeout=float(self.timeout))

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
        """Constructs inference options for Ollama respecting acceleration configuration."""
        options: Dict[str, Any] = {
            "temperature": temperature if temperature is not None else self.temperature,
            "num_predict": max_tokens or self.max_tokens,
            "num_ctx": getattr(settings, "NUM_CTX", 3072),
        }

        if getattr(settings, "ENABLE_ACCELERATION", False):
            device = getattr(settings, "ACCELERATION_DEVICE", "cpu").lower()
            threads = getattr(settings, "ACCELERATION_THREADS", 4)
            options["num_thread"] = max(1, threads)
            if device in ("gpu", "cuda"):
                options["num_gpu"] = 99
            elif device == "cpu":
                options["num_gpu"] = 0
            elif device == "auto":
                options["num_gpu"] = 99 if getattr(settings, "is_gpu_available", lambda: False)() else 0
        else:
            options["num_thread"] = 1
            options["num_gpu"] = 0

        return options

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Streams generation tokens chunk-by-chunk from the local Ollama LLM.
        
        Yields periodic empty string heartbeats during long prompt evaluation to prevent socket read timeouts.
        """
        if not prompt.strip():
            return

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        options = self._build_options(temperature, max_tokens)

        token_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        error_holder: List[Exception] = []

        def _worker():
            try:
                stream = self._client.chat(
                    model=self.model_name,
                    messages=messages,
                    options=options,
                    stream=True,
                )
                for chunk in stream:
                    if stop_event.is_set():
                        break
                    token = ""
                    if hasattr(chunk, "message"):
                        token = chunk.message.content or ""
                    elif isinstance(chunk, dict) and "message" in chunk:
                        token = chunk["message"].get("content", "")
                    if token:
                        token_queue.put(("token", token))
            except Exception as exc:
                error_holder.append(exc)
            finally:
                token_queue.put(("done", None))

        try:
            logger.debug("Starting streaming generation with Ollama (model: %s)", self.model_name)
            worker_thread = threading.Thread(target=_worker, daemon=True)
            worker_thread.start()

            while True:
                try:
                    item_type, data = token_queue.get(timeout=3.0)
                    if item_type == "token":
                        yield data
                    elif item_type == "done":
                        break
                except queue.Empty:
                    if error_holder:
                        break
                    # Yield heartbeat sentinel to keep HTTP streaming socket alive during prompt eval
                    yield ""

            if error_holder:
                exc = error_holder[0]
                err_msg = str(exc).lower()
                if "connection" in err_msg or "connect" in err_msg:
                    raise OllamaConnectionError(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?") from exc
                if "not found" in err_msg:
                    raise ModelNotAvailableError(
                        f"Ollama model '{self.model_name}' is not installed. Run 'ollama pull {self.model_name}'."
                    ) from exc
                raise RAGException(f"Ollama streaming inference error: {exc}") from exc

        finally:
            stop_event.set()
