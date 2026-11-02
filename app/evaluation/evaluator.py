"""RAG evaluation orchestrator and reporting engine."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
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


@dataclass
class EvaluationSample:
    question: str
    expected_answer: Optional[str] = None
    relevant_chunk_ids: List[str] = field(default_factory=list)
    relevant_filenames: List[str] = field(default_factory=list)
    expected_to_abstain: bool = False


@dataclass
class EvaluationResult:
    sample_count: int
    precision_at_5: float
    recall_at_5: float
    hit_rate_at_5: float
    mean_reciprocal_rank: float
    average_faithfulness: float
    abstention_accuracy: float
    details: List[Dict[str, Any]] = field(default_factory=list)


class RAGEvaluator:
    """Executes quality benchmarks on retrieval, generation, faithfulness, and abstention."""

    @classmethod
    def evaluate_sample(
        cls,
        sample: EvaluationSample,
        retrieved_ids: List[str],
        generated_answer: str,
        citations: List[dict],
        context_text: str,
        is_abstained: bool,
    ) -> Dict[str, Any]:
        """Evaluates an individual test query against ground truth expectations."""
        relevant_set = set(sample.relevant_chunk_ids)

        p5 = precision_at_k(retrieved_ids, relevant_set, k=5) if relevant_set else 1.0
        r5 = recall_at_k(retrieved_ids, relevant_set, k=5) if relevant_set else 1.0
        hr5 = hit_rate_at_k(retrieved_ids, relevant_set, k=5) if relevant_set else 1.0
        mrr = mean_reciprocal_rank(retrieved_ids, relevant_set) if relevant_set else 1.0

        faithfulness = context_faithfulness(generated_answer, context_text)
        abstention_correct = (is_abstained == sample.expected_to_abstain)

        f1 = 0.0
        if sample.expected_answer and not is_abstained:
            f1 = token_f1_score(generated_answer, sample.expected_answer)

        return {
            "question": sample.question,
            "precision_at_5": p5,
            "recall_at_5": r5,
            "hit_rate_at_5": hr5,
            "mrr": mrr,
            "faithfulness": faithfulness,
            "abstention_correct": abstention_correct,
            "token_f1": f1,
        }
