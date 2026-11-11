"""Answer validation and hallucination prevention using deterministic grounding checks."""

import re
from typing import Any, Dict, List, Set, Tuple
from app.core.logging import logger
from app.retrieval.base import RetrievedChunk


ABSTENTION_MESSAGE = "The available documents do not contain sufficient information to answer this question accurately."


class AnswerValidator:
    """Performs deterministic hallucination detection, numerical grounding, and abstention verification."""

    # Regex to detect numbers, percentages, currency, and dates
    _NUMERICAL_REGEX = re.compile(r"\b(?:\$|€|£)?\d+(?:[\.,]\d+)?%?\b")

    # Patterns indicating LLM abstention
    _ABSTENTION_TRIGGERS = [
        re.compile(r"do\s+not\s+contain\s+sufficient\s+information", re.IGNORECASE),
        re.compile(r"insufficient\s+information", re.IGNORECASE),
        re.compile(r"not\s+mentioned\s+in\s+the\s+provided\s+context", re.IGNORECASE),
        re.compile(r"context\s+does\s+not\s+provide", re.IGNORECASE),
        re.compile(r"no\s+mention\s+of", re.IGNORECASE),
    ]

    @classmethod
    def is_abstention(cls, text: str) -> bool:
        """Detects whether text is purely or predominantly an abstention response."""
        clean = text.strip()
        has_citations = "[source:" in clean.lower()
        has_bullets = any(line.strip().startswith(("-", "*", "1.", "2.", "•")) for line in clean.split("\n"))

        matches_trigger = any(pattern.search(clean) for pattern in cls._ABSTENTION_TRIGGERS)
        if not matches_trigger:
            return False

        # If it matches trigger and has no citations and is short, it is a pure abstention
        if len(clean) < 180 and not has_citations:
            return True

        # If it contains citations or bullets or substantive content, it is answering supported portions
        if has_citations or has_bullets or len(clean) > 250:
            return False

        return True

    @classmethod
    def check_retrieval_sufficiency(
        cls,
        chunks: List[RetrievedChunk],
        min_score: float = 0.20,
    ) -> bool:
        """Determines if the retrieved candidate pool has sufficient relevance."""
        if not chunks:
            return False
        # Best chunk must meet minimum relevance threshold
        best_score = max(c.score for c in chunks)
        if best_score < min_score:
            logger.info("Retrieval sufficiency check failed: top score %.4f < %.4f", best_score, min_score)
            return False
        return True

    @classmethod
    def validate_numerical_grounding(
        cls,
        answer: str,
        context_text: str,
    ) -> Tuple[bool, List[str]]:
        """Verifies that numbers/percentages in the answer originate from the context."""
        # Strip citation tags (e.g. [Source: doc.pdf, Page: 18]) before checking factual numbers
        clean_answer = re.sub(r"\[Source:[^\]]+\]", "", answer, flags=re.IGNORECASE)
        answer_nums = set(cls._NUMERICAL_REGEX.findall(clean_answer))
        context_nums = set(cls._NUMERICAL_REGEX.findall(context_text))

        # Filter out common small integers like 1, 2, 3 (often from bullet points)
        significant_nums = {n for n in answer_nums if not (n.isdigit() and int(n) < 10)}
        unsupported = [n for n in significant_nums if n not in context_nums]

        is_grounded = len(unsupported) == 0
        if not is_grounded:
            logger.warning("Detected potentially ungrounded numbers in answer: %s", unsupported)

        return is_grounded, unsupported

    @classmethod
    def clean_meta_language(cls, text: str) -> str:
        """Sanitizes generated answer text by removing meta-language, internal block references,
        robotic preambles, and repetitive inline headings.
        """
        if not text:
            return ""

        s = text.strip()

        # 1. Strip trailing raw citation dumps like 'Citation Tags: hees101.pdf...' or 'References: [1] ...'
        s = re.sub(
            r"(?i)(?:\n+|\s+)(?:citation\s+tags?|references?|sources?)\s*:.*$",
            "",
            s,
        ).strip()

        # 2. Strip robotic opening preambles at the start
        s = re.sub(
            r"^(?:based\s+on|according\s+to|per)\s+(?:the\s+)?(?:retrieved|provided|available)\s+(?:document\s+)?(?:context|information|excerpts|data)[,:]?\s*",
            "",
            s,
            flags=re.IGNORECASE,
        )
        if s and s[0].islower():
            s = s[0].upper() + s[1:]

        # 3. Strip embedded meta phrases like ', supported by the retrieved document context:'
        s = re.sub(
            r",?\s*(?:as\s+)?(?:supported\s+by|based\s+on|according\s+to)\s+the\s+(?:retrieved|provided)\s+(?:document\s+)?(?:context|information|excerpts)[,:]?",
            ":",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(r":\s*:", ":", s)

        # 4. 'As stated in Context Block X', 'According to Context Block X', 'In Context Block X', etc.
        s = re.sub(
            r"\b(?:as\s+(?:stated|mentioned|noted|discussed|indicated|described)\s+in|according\s+to|in|per|from)\s+(?:context\s+block|document\s+excerpt|excerpt)\s+\d+[,:]?\s*",
            "",
            s,
            flags=re.IGNORECASE,
        )

        # 5. 'Context Block X (also )?(emphasizes|states|notes|highlights|discusses|mentions|points out|explains)... stating/noting:'
        s = re.sub(
            r"\b(?:context\s+block|document\s+excerpt|excerpt)\s+\d+\s+(?:also\s+)?(?:states|notes|highlights|emphasizes|discusses|mentions|points\s+out|explains|indicates|shows)[^,\n:]*[,:]?\s*(?:stating|noting|explaining|indicating)?[,:]?\s*",
            "",
            s,
            flags=re.IGNORECASE,
        )

        # 6. Any leftover 'Context Block \d+' or 'Document Excerpt \d+'
        s = re.sub(r"\b(?:context\s+block|document\s+excerpt)\s+\d+[,:]?\s*", "", s, flags=re.IGNORECASE)

        # 7. Clean meta phrases in body text and convert to natural documentation phrasing
        s = re.sub(
            r"(?i)\bthe\s+retriev(?:al|ed)\s+(?:document\s+)?context\s+(?:provides|indicates|shows|outlines)\s+",
            "the available documentation outlines ",
            s,
        )
        s = re.sub(
            r"(?i)\bbased\s+on\s+the\s+retriev(?:al|ed)\s+(?:document\s+)?context[,:]?\s*",
            "According to the available documentation, ",
            s,
        )
        s = re.sub(
            r"(?i)\bthe\s+retriev(?:al|ed)\s+(?:document\s+)?context\b",
            "the available documentation",
            s,
        )
        s = re.sub(
            r"(?i)\bretriev(?:al|ed)\s+(?:document\s+)?context\b",
            "available documentation",
            s,
        )
        s = re.sub(
            r"(?i)\bthe\s+provided\s+context\b",
            "the available documentation",
            s,
        )

        # 8. Unquote text following bullet colon if quotes were wrapped around the excerpt
        s = re.sub(r'(:\s*)"([^"]+)"', r"\1\2", s)

        # 9. Deduplicate repeating inline titles (e.g. '• Title: Title: text')
        lines = []
        for line in s.split("\n"):
            m = re.match(r"^(\s*[-*•\d.]+\s*)([^:\n]+):\s*\2\s*:\s*(.*)$", line, re.IGNORECASE)
            if m:
                lines.append(f"{m.group(1)}{m.group(2)}: {m.group(3)}")
            else:
                lines.append(line)
        s = "\n".join(lines)

        # 10. Clean redundant inline spaces and leading commas
        s = re.sub(r"[ \t]{2,}", " ", s)
        s = re.sub(r"^[,\s]+", "", s)
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        return s.strip()

    @classmethod
    def format_professional_markdown(cls, text: str) -> str:
        """Formats raw or squashed LLM answers into a clean, executive document layout with
        proper markdown lists, bold headings, and breathing room between sections.
        """
        if not text:
            return ""

        # Protect fenced code blocks from formatting modifications
        parts = re.split(r"(```[\s\S]*?```)", text)
        formatted_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                # Inside code block: preserve verbatim
                formatted_parts.append(part)
                continue

            s = part

            # 0. Elevate a standalone unpunctuated title line (<= 8 words) followed by text to a markdown heading:
            # e.g. "Ecosystem Functions Serving Humans\nThe available documentation..." -> "### Ecosystem Functions Serving Humans\n\nThe available..."
            first_line_match = re.match(r"^([A-Z][a-zA-Z0-9\s]{2,60})\n+([a-zA-Z0-9])", s)
            if first_line_match:
                title = first_line_match.group(1).strip()
                rest = s[first_line_match.start(2):]
                if rest and rest[0].islower():
                    rest = rest[0].upper() + rest[1:]
                if not title.endswith((".", ":", ";", ",", "?", "!")) and len(title.split()) <= 8 and not title.startswith("#"):
                    s = f"### {title}\n\n" + rest

            # 0b. If text starts with bold title like **Title**\nText, promote to ### Title\n\nText
            s = re.sub(r"^\*{2}([^\n\*]+)\*{2}\n+([^\n])", r"### \1\n\n\2", s)

            # 1. Fix inline first bullet stuck to colon introductory phrase:
            # e.g. "Here are five examples: Filtering Water: ecosystems..." -> "Here are five examples:\n\n- Filtering Water: ecosystems..."
            s = re.sub(
                r"(:\s*)(\*{0,2}[A-Z][a-zA-Z0-9\s]{2,40}\*{0,2}\s*[:\-]\s*[a-zA-Z])",
                r":\n\n- \2",
                s,
            )

            # 2. Convert unicode bullet symbols (•, ●, ▪, etc.) squashed inline or with whitespace into proper markdown list items
            s = re.sub(r"(?:\s*[\r\n]+\s*|\s+)[•●▪]\s*", "\n\n- ", s)
            # If starts with a bullet symbol
            s = re.sub(r"^[•●▪]\s*", "- ", s)

            # 3. Split inline numbered lists squashed into a single line:
            # e.g. "Intro: 1. Item one. 2. Item two." -> "Intro:\n\n1. Item one.\n\n2. Item two."
            s = re.sub(r"(:\s*|[\.\]\)]\s+)(\d+\.\s+)", r"\1\n\n\2", s)

            # 4. Bold category/term heading before colon in list items:
            # e.g. "- Renewable Resource Management: Humans can..." -> "- **Renewable Resource Management**: Humans can..."
            def _bold_item_heading(match):
                prefix = match.group(1)  # "- " or "1. "
                heading = match.group(2).strip()
                rest = match.group(3)
                # Avoid re-bolding if already bolded
                if heading.startswith("**") and heading.endswith("**"):
                    return f"{prefix}{heading}: {rest}"
                # Only bold concise title phrases (<= 8 words)
                if len(heading.split()) <= 8:
                    return f"{prefix}**{heading}**: {rest}"
                return match.group(0)

            s = re.sub(
                r"(?m)^(\s*(?:-\s+|\d+\.\s+))([^:\n\*]+?)\s*:\s*([^\n]+)$",
                _bold_item_heading,
                s,
            )

            # 5. Ensure a blank line before the first list item if preceded by an introductory sentence
            s = re.sub(r"([^\n])\n(- |\d+\. )", r"\1\n\n\2", s)

            # 6. Ensure headers (### Section) have blank lines before and after
            s = re.sub(r"([^\n])\n(#{1,4}\s+[^\n]+)", r"\1\n\n\2", s)
            s = re.sub(r"(#{1,4}\s+[^\n]+)\n([^\n#])", r"\1\n\n\2", s)

            # 7. Ensure concluding paragraphs after a list have a blank line
            lines = s.split("\n")
            processed = []
            in_list = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                    in_list = True
                    processed.append(line)
                elif in_list and stripped and not stripped.startswith(("#", "-", "*")):
                    in_list = False
                    processed.append("")
                    processed.append(line)
                else:
                    if not stripped:
                        in_list = False
                    processed.append(line)
            s = "\n".join(processed)

            # 8. Normalize paragraph spacing (at most 2 newlines)
            s = re.sub(r"\n{3,}", "\n\n", s)

            formatted_parts.append(s)

        res = "".join(formatted_parts).strip()
        return res

    @classmethod
    def validate_answer(
        cls,
        question: str,
        answer: str,
        chunks: List[RetrievedChunk],
    ) -> Dict[str, Any]:
        """Runs the complete answer validation and evidence grounding evaluation."""
        warnings: List[str] = []

        # 1. Did the assistant abstain?
        if cls.is_abstention(answer):
            return {
                "is_grounded": True,
                "should_abstain": True,
                "validated_answer": ABSTENTION_MESSAGE,
                "warnings": [],
            }

        # 2. Check retrieval sufficiency
        if not cls.check_retrieval_sufficiency(chunks):
            logger.info("Abstaining: Retrieved documents lack sufficient evidence for question: '%s'", question)
            return {
                "is_grounded": False,
                "should_abstain": True,
                "validated_answer": ABSTENTION_MESSAGE,
                "warnings": ["Insufficient retrieval score to guarantee grounded answer."],
            }

        # If the model provided substantive content but opened with the canned abstention sentence, clean it
        cleaned_body = re.sub(r"^" + re.escape(ABSTENTION_MESSAGE) + r"\s*", "", answer).strip()
        if cleaned_body:
            answer = cleaned_body

        # Clean meta-language, internal block references, robotic preambles, and duplicate headings
        answer = cls.clean_meta_language(answer)
        # Format into clean, executive document structure with proper markdown hierarchy and spacing
        answer = cls.format_professional_markdown(answer)

        # 3. Check numerical grounding
        combined_context = " ".join(c.text for c in chunks)
        nums_ok, unsupported_nums = cls.validate_numerical_grounding(answer, combined_context)
        if not nums_ok:
            warnings.append(f"Answer contains numbers not found in retrieved context: {unsupported_nums}")

        return {
            "is_grounded": nums_ok,
            "should_abstain": False,
            "validated_answer": answer,
            "warnings": warnings,
        }
