"""RAG pipeline evaluation framework."""

from app.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
)
from app.evaluation.generation_metrics import (
    token_f1_score,
    context_faithfulness,
    citation_correctness,
)
from app.evaluation.evaluator import (
    EvaluationSample,
    EvaluationResult,
    RAGEvaluator,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "hit_rate_at_k",
    "mean_reciprocal_rank",
    "token_f1_score",
    "context_faithfulness",
    "citation_correctness",
    "EvaluationSample",
    "EvaluationResult",
    "RAGEvaluator",
]
