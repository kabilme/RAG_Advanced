"""Optional OCR processor for scanned documents and images using pytesseract."""

import shutil
from typing import Optional
from PIL import Image
from app.core.config import settings
from app.core.logging import logger


class OCRProcessor:
    """Manages local OCR operations via Tesseract if configured and available."""

    _tesseract_checked: bool = False
    _tesseract_available: bool = False

    @classmethod
    def is_available(cls) -> bool:
        """Checks if OCR is enabled in settings and tesseract binary is in system PATH."""
        if not settings.ENABLE_OCR:
            return False

        if not cls._tesseract_checked:
            cls._tesseract_checked = True
            try:
                import pytesseract
                # Check if tesseract binary exists in PATH
                tesseract_path = shutil.which("tesseract")
                if tesseract_path:
                    cls._tesseract_available = True
                    logger.info("Local Tesseract OCR engine detected at: %s", tesseract_path)
                else:
                    cls._tesseract_available = False
                    logger.debug("Tesseract binary not found in system PATH. OCR is disabled.")
            except ImportError:
                cls._tesseract_available = False
                logger.debug("pytesseract library not available.")

        return cls._tesseract_available

    @classmethod
    def extract_text_from_image(cls, image: Image.Image) -> Optional[str]:
        """Extracts text from a PIL Image using Tesseract OCR.
        
        Returns:
            Extracted text if successful, or None if OCR is unavailable/failed.
        """
        if not cls.is_available():
            return None

        try:
            import pytesseract
            text = pytesseract.image_to_string(image)
            return text.strip() if text else None
        except Exception as exc:
            logger.warning("OCR text extraction failed: %s", exc)
            return None
