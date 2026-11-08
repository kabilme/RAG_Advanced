"""Multimodal vision processor for extracting, analyzing, and describing diagrams, charts, and figures."""

import hashlib
import io
from typing import Any, Dict, List, Optional
from PIL import Image
import ollama

from app.core.config import settings
from app.core.logging import logger

DEFAULT_VISION_PROMPT = (
    "Analyze this image from a document. "
    "If it is a chart, graph, technical diagram, flowchart, map, architecture diagram, or meaningful data visual: "
    "provide a concise factual summary describing: "
    "1. Type of visual (e.g. bar chart, line graph, flowchart, architecture diagram). "
    "2. Key title, labels, axes, metrics, components, and relationships. "
    "3. The primary data findings, trends, or workflow steps shown. "
    "If it is just a logo, banner, page border, divider, icon, or decorative graphic, reply strictly with 'SKIP'."
)


class VisionProcessor:
    """Processes document images and generates semantic descriptions for diagrams and charts using Ollama."""

    _checked: bool = False
    _available: bool = False
    _cache: Dict[str, Optional[str]] = {}

    @classmethod
    def reset_state(cls) -> None:
        """Resets cached availability and image descriptions (useful for testing or config updates)."""
        cls._checked = False
        cls._available = False
        cls._cache.clear()

    @classmethod
    def is_available(cls) -> bool:
        """Verifies if vision ingestion is enabled and the configured vision model is available in Ollama."""
        if not getattr(settings, "ENABLE_VISION", False):
            return False

        if not cls._checked:
            cls._checked = True
            try:
                client = ollama.Client(host=settings.OLLAMA_BASE_URL)
                resp = client.list()
                installed = []
                if hasattr(resp, "models"):
                    for m in resp.models:
                        name = getattr(m, "model", None) or getattr(m, "name", "")
                        installed.append(name.lower())
                elif isinstance(resp, dict) and "models" in resp:
                    for m in resp["models"]:
                        installed.append(m.get("name", m.get("model", "")).lower())

                target_model = getattr(settings, "VISION_MODEL", "moondream").lower()
                target_prefix = target_model.split(":")[0]

                # Check if configured model or prefix exists
                cls._available = any(
                    target_model in m or m.startswith(f"{target_prefix}:") or m == target_prefix
                    for m in installed
                )
                if cls._available:
                    logger.info("Local Ollama vision model '%s' detected and enabled.", target_model)
                else:
                    logger.info(
                        "Vision model '%s' not detected in Ollama. Available: %s. Vision ingestion disabled.",
                        target_model, installed
                    )
            except Exception as exc:
                cls._available = False
                logger.debug("Ollama vision availability check skipped: %s", exc)

        return cls._available

    @classmethod
    def prepare_image_for_vision(cls, image_bytes: bytes) -> Optional[bytes]:
        """Validates and normalizes raw image bytes into a clean 24-bit RGB PNG byte stream.

        Filters out:
        - Files below minimum byte size threshold
        - Undersized images (< min_dimension)
        - Extreme aspect ratios (e.g., thin 1px dividers or borders)
        - Completely solid-color blank images/masks (e.g. all black/white transparency layers)

        Converts CMYK, indexed palette, grayscale, and alpha channels into standard
        RGB PNG format, preventing 400 Bad Request decoding errors in Ollama's stb_image parser.
        """
        if not image_bytes:
            return None

        min_size = getattr(settings, "VISION_MIN_IMAGE_SIZE", 4096)
        min_dim = getattr(settings, "VISION_MIN_DIMENSION", 100)

        if len(image_bytes) < min_size:
            return None

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                w, h = img.size
                if w < min_dim or h < min_dim:
                    return None

                # Reject extreme aspect ratios like thin separator bars
                aspect = max(w, h) / max(1, min(w, h))
                if aspect > 15.0:
                    return None

                # Reject solid-color blank images or transparency masks
                extrema = img.getextrema()
                if img.mode in ("1", "L") and isinstance(extrema, tuple):
                    if extrema[0] == extrema[1]:
                        return None
                elif img.mode in ("RGB", "RGBA", "CMYK"):
                    if isinstance(extrema, tuple) and all(isinstance(b, tuple) and b[0] == b[1] for b in extrema):
                        return None

                # Convert to standard 24-bit RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Downscale excessively large scans (> 1600px) to prevent VRAM spikes
                max_edge = 1600
                if max(w, h) > max_edge:
                    scale = max_edge / max(w, h)
                    new_size = (int(w * scale), int(h * scale))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                return buf.getvalue()
        except Exception as exc:
            logger.debug("Image normalization skipped: %s", exc)
            return None

    @classmethod
    def is_meaningful_image(cls, image_bytes: bytes) -> bool:
        """Determines if an image is large and dimensional enough to contain charts or diagrams."""
        return cls.prepare_image_for_vision(image_bytes) is not None

    @classmethod
    def describe_image(
        cls,
        image_bytes: bytes,
        prompt: Optional[str] = None,
        custom_client: Optional[Any] = None,
    ) -> Optional[str]:
        """Analyzes an image with the local vision model and returns a structured textual description."""
        if not image_bytes:
            return None

        # Normalize and filter image (converts CMYK/masks to standard RGB PNG)
        prepared_png = cls.prepare_image_for_vision(image_bytes)
        if not prepared_png:
            return None

        # Check sha256 cache using prepared PNG bytes
        img_hash = hashlib.sha256(prepared_png).hexdigest()
        if img_hash in cls._cache:
            return cls._cache[img_hash]

        if not cls.is_available() and custom_client is None:
            return None

        analysis_prompt = prompt or DEFAULT_VISION_PROMPT
        model_name = getattr(settings, "VISION_MODEL", "moondream")

        try:
            client = custom_client or ollama.Client(host=settings.OLLAMA_BASE_URL)
            logger.debug("Dispatching normalized PNG to Ollama vision model '%s' (hash: %s)", model_name, img_hash[:8])
            resp = client.generate(
                model=model_name,
                prompt=analysis_prompt,
                images=[prepared_png],
                options={"temperature": 0.1, "num_predict": 300},
            )

            result_text = resp.response.strip() if hasattr(resp, "response") else str(resp).strip()

            # Handle skip response
            if not result_text or result_text.upper().startswith("SKIP"):
                cls._cache[img_hash] = None
                return None

            cls._cache[img_hash] = result_text
            logger.info("Vision model extracted description for figure (%d chars): %s...", len(result_text), result_text[:60])
            return result_text

        except Exception as exc:
            err_str = str(exc)
            if "unknown model architecture" in err_str or "mllama" in err_str:
                logger.error(
                    "Ollama vision model '%s' failed: The running Ollama server version does not support the 'mllama' architecture. "
                    "To fix this, update Ollama to v0.4.0+ from https://ollama.com, or switch VISION_MODEL to a widely compatible vision model like 'moondream' (1.8B, fast) or 'llava' (7B). "
                    "Temporarily disabling vision processing for this session.",
                    model_name
                )
                cls._available = False
            else:
                logger.warning("Vision analysis failed for image: %s", exc)
            cls._cache[img_hash] = None
            return None

