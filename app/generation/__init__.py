"""Generation package orchestrating local LLM inference, grounded prompts, and citations."""

from app.generation.ollama_client import OllamaClient
from app.generation.prompt_builder import PromptBuilder, RAG_SYSTEM_PROMPT
from app.generation.citation_manager import CitationManager, CitationItem
from app.generation.answer_validator import AnswerValidator, ABSTENTION_MESSAGE
from app.generation.rag_chain import RAGChain, RAGResponse

__all__ = [
    "OllamaClient",
    "PromptBuilder",
    "RAG_SYSTEM_PROMPT",
    "CitationManager",
    "CitationItem",
    "AnswerValidator",
    "ABSTENTION_MESSAGE",
    "RAGChain",
    "RAGResponse",
]
