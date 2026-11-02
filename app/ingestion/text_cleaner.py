"""Text cleaning, normalization, and untrusted input sanitization."""

import re
import unicodedata


class TextCleaner:
    """Cleans, normalizes, and sanitizes extracted document text."""

    # Matches script tags, inline javascript, and dangerous tags
    _HTML_TAG_PATTERN = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
    _STYLE_TAG_PATTERN = re.compile(r"<style[^>]*>.*?</style>", re.IGNORECASE | re.DOTALL)
    _INLINE_JS_PATTERN = re.compile(r"javascript:[^\s\"'>]+", re.IGNORECASE)

    # Patterns for potential prompt-injection triggers often embedded in malicious documents
    _INJECTION_MARKERS = [
        re.compile(r"system\s*prompt\s*override", re.IGNORECASE),
        re.compile(r"(disregard|ignore)\s+(all\s+)?(previous|prior)\s+(instructions|commands|rules|prompts)", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
    ]

    @classmethod
    def clean(cls, text: str, preserve_layout: bool = True) -> str:
        """Normalizes and cleans text extracted from documents.
        
        Args:
            text: Raw extracted string.
            preserve_layout: Whether to maintain meaningful line breaks.
            
        Returns:
            Cleaned and normalized text.
        """
        if not text:
            return ""

        # 1. Unicode normalization (NFKC)
        normalized = unicodedata.normalize("NFKC", text)

        # 2. Strip embedded dangerous scripts or html wrappers
        normalized = cls._HTML_TAG_PATTERN.sub(" ", normalized)
        normalized = cls._STYLE_TAG_PATTERN.sub(" ", normalized)
        normalized = cls._INLINE_JS_PATTERN.sub(" ", normalized)

        # 3. Replace non-printable control characters except tab and newline
        cleaned_chars = []
        for char in normalized:
            category = unicodedata.category(char)
            # Allow Cc (control) only if \n, \r, \t
            if category.startswith("C") and char not in ("\n", "\r", "\t"):
                cleaned_chars.append(" ")
            else:
                cleaned_chars.append(char)
        normalized = "".join(cleaned_chars)

        # 4. Normalize newlines
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

        # 5. Clean up excessive spaces while preserving layout
        if preserve_layout:
            # Replace 3 or more consecutive newlines with 2 newlines (preserve paragraphs)
            normalized = re.sub(r"\n{3,}", "\n\n", normalized)
            # Replace multiple horizontal spaces or tabs with a single space
            normalized = re.sub(r"[ \t]+", " ", normalized)
            # Remove trailing/leading spaces on each line
            lines = [line.strip() for line in normalized.split("\n")]
            normalized = "\n".join(lines)
        else:
            normalized = re.sub(r"\s+", " ", normalized)

        return normalized.strip()

    @classmethod
    def detect_injection_attempts(cls, text: str) -> bool:
        """Flags whether text contains obvious prompt injection directives."""
        if not text:
            return False
        for marker in cls._INJECTION_MARKERS:
            if marker.search(text):
                return True
        return False
