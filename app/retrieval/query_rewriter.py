"""Query understanding, classification, follow-up resolution, and subquery decomposition."""

import re
from typing import Any, Dict, List, Optional
from app.core.logging import logger


class QueryRewriter:
    """Processes, classifies, and resolves conversational queries for optimal retrieval."""

    _PRONOUNS_PATTERN = re.compile(
        r"\b(it|its|they|them|their|this|that|these|those)\b", re.IGNORECASE
    )
    _FOLLOWUP_TRIGGERS = re.compile(
        r"^(and\b|what\s+about\b|how\s+about\b|why\b|why\s+not\b|tell\s+me\s+more\b)", re.IGNORECASE
    )

    @classmethod
    def classify_query(cls, query: str) -> str:
        """Classifies query intent into defined RAG question categories."""
        q = query.strip().lower()

        if any(term in q for term in ["compare", "difference between", "versus", "vs.", "better than"]):
            return "comparative"
        if any(term in q for term in ["summarize", "summary", "overview", "briefly describe", "recap"]):
            return "summarization"
        if any(term in q for term in ["how many", "how much", "percentage", "total", "calculate", "average", "cost", "revenue", "count"]):
            return "numerical"
        if any(term in q for term in ["what is", "who is", "where is", "when did", "define", "name"]):
            return "factual"
        if any(term in q for term in ["explain", "describe", "how does", "why does", "details"]):
            return "descriptive"
        if " and " in q and ("?" in q or len(q.split()) > 10):
            return "multi_part"

        return "factual"

    @classmethod
    def resolve_followup(cls, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Resolves coreferences and pronoun bindings using previous conversation context."""
        if not conversation_history:
            return query

        clean_query = query.strip()
        has_pronoun = bool(cls._PRONOUNS_PATTERN.search(clean_query))
        is_followup_trigger = bool(cls._FOLLOWUP_TRIGGERS.search(clean_query))

        if not (has_pronoun or is_followup_trigger):
            return clean_query

        # Look at the most recent user question in history
        last_user_query = None
        for turn in reversed(conversation_history):
            if turn.get("role") == "user" and turn.get("content"):
                last_user_query = turn["content"].strip()
                break

        if not last_user_query:
            return clean_query

        # Extract subject from last query (strip question words)
        subject_candidate = re.sub(
            r"^(what\s+is\s+|what\s+are\s+|how\s+does\s+|explain\s+|tell\s+me\s+about\s+)",
            "",
            last_user_query,
            flags=re.IGNORECASE,
        ).rstrip("?.,")

        # Resolve pronoun or prepend context if appropriate
        if "its" in clean_query.lower() or "it" in clean_query.lower():
            resolved = re.sub(
                r"\b(it's|its|it)\b",
                f"{subject_candidate}'s" if "its" in clean_query.lower() else subject_candidate,
                clean_query,
                flags=re.IGNORECASE,
            )
            logger.info("Resolved coreference: '%s' -> '%s'", clean_query, resolved)
            return resolved

        if is_followup_trigger:
            resolved = f"{clean_query} regarding {subject_candidate}"
            logger.info("Expanded follow-up context: '%s' -> '%s'", clean_query, resolved)
            return resolved

        return clean_query

    @classmethod
    def decompose_query(cls, query: str) -> List[str]:
        """Decomposes compound or multi-part questions into distinct sub-queries."""
        clean = query.strip()
        # Look for conjunctions separating question clauses
        parts = re.split(r"\s+and\s+(?=what|how|why|where|when|can|does|is|are|will)", clean, flags=re.IGNORECASE)
        if len(parts) > 1:
            subqueries = [p.strip() for p in parts if p.strip()]
            logger.debug("Decomposed query '%s' into: %s", clean, subqueries)
            return subqueries

        return [clean]
