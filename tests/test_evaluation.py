"""Unit tests for RAG evaluation framework."""

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
from app.evaluation.evaluator import EvaluationSample, RAGEvaluator


def test_retrieval_metrics():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc2", "doc5"}

    # Precision@3 = 1 / 3
    assert abs(precision_at_k(retrieved, relevant, k=3) - (1 / 3)) < 1e-4

    # Recall@3 = 1 / 2
    assert abs(recall_at_k(retrieved, relevant, k=3) - 0.5) < 1e-4

    # Hit Rate@3 = 1.0 (doc2 is in top 3)
    assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0

    # MRR: first relevant is at rank 2 -> 1/2 = 0.5
    assert abs(mean_reciprocal_rank(retrieved, relevant) - 0.5) < 1e-4


def test_generation_metrics():
    # Token F1
    pred = "Solar panels last 25 to 30 years."
    gt = "Expected lifespan is 25 to 30 years."
    f1 = token_f1_score(pred, gt)
    assert f1 > 0.5

    # Context faithfulness
    context = "Solar panels last 25 to 30 years in outdoor conditions."
    answer = "Solar panels last 25 to 30 years."
    faith = context_faithfulness(answer, context)
    assert faith > 0.8


def test_rag_evaluator_sample():
    sample = EvaluationSample(
        question="What is the camel's name?",
        expected_to_abstain=True,
    )
    result = RAGEvaluator.evaluate_sample(
        sample=sample,
        retrieved_ids=[],
        generated_answer="The available documents do not contain sufficient information to answer this question accurately.",
        citations=[],
        context_text="",
        is_abstained=True,
    )
    assert result["abstention_correct"] is True
