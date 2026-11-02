"""Evaluation metrics for answer quality, faithfulness, and citation correctness."""

import re
from typing import List, Set


def token_f1_score(prediction: str, ground_truth: str) -> float:
    """Computes word-level token overlap F1 score between prediction and ground truth."""
    pred_tokens = re.findall(r"\w+", prediction.lower())
    gt_tokens = re.findall(r"\w+", ground_truth.lower())

    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common = sum((pred_tokens.count(t) for t in set(gt_tokens) if t in pred_tokens))
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def context_faithfulness(answer: str, context_text: str) -> float:
    """Estimates the proportion of answer content words supported by the retrieved context."""
    answer_words = [w for w in re.findall(r"\w+", answer.lower()) if len(w) > 2]
    if not answer_words:
        return 1.0

    context_words = set(re.findall(r"\w+", context_text.lower()))
    supported = sum(1 for w in answer_words if w in context_words)
    return supported / len(answer_words)


def citation_correctness(citations: List[dict], retrieved_filenames: Set[str]) -> float:
    """Calculates the ratio of citations matching verified retrieved filenames."""
    if not citations:
        return 1.0
    valid = sum(1 for c in citations if c.get("filename", "").lower() in retrieved_filenames or c.get("is_valid", False))
    return valid / len(citations)
