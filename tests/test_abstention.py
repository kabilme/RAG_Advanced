"""Unit tests for deterministic abstention and evidence grounding rules."""

import pytest
from app.retrieval.base import RetrievedChunk
from app.generation.answer_validator import AnswerValidator, ABSTENTION_MESSAGE


@pytest.fixture
def sample_context_chunk():
    return [
        RetrievedChunk(
            chunk_id="chunk-merchant",
            document_id="doc-camel",
            text="The merchant traveled across the Thar desert with three camels carrying silk and spices.",
            score=0.85,
            rank=1,
            retrieval_method="hybrid",
            filename="desert_tales.txt",
        )
    ]


def test_abstention_detection():
    assert AnswerValidator.is_abstention("The available documents do not contain sufficient information to answer.") is True
    assert AnswerValidator.is_abstention("This is not mentioned in the provided context.") is True
    assert AnswerValidator.is_abstention("Solar power uses photovoltaic cells.") is False


def test_retrieval_insufficiency_abstention():
    # Empty chunks -> insufficient
    assert AnswerValidator.check_retrieval_sufficiency([]) is False

    # Chunks with very low relevance score -> insufficient
    low_score_chunks = [
        RetrievedChunk(
            chunk_id="c_low",
            document_id="doc_x",
            text="Irrelevant text.",
            score=0.08,
            rank=1,
            retrieval_method="hybrid",
        )
    ]
    assert AnswerValidator.check_retrieval_sufficiency(low_score_chunks, min_score=0.20) is False

    # Chunks with high relevance score -> sufficient
    good_chunks = [
        RetrievedChunk(
            chunk_id="c_good",
            document_id="doc_y",
            text="Relevant text.",
            score=0.85,
            rank=1,
            retrieval_method="hybrid",
        )
    ]
    assert AnswerValidator.check_retrieval_sufficiency(good_chunks, min_score=0.20) is True


def test_unsupported_question_abstention(sample_context_chunk):
    # Test case matching Section 19 of prompt:
    # Question: "What is the name of the merchant's camel?"
    # The document says "three camels carrying silk" but never names the camel!
    llm_abstention = (
        "The available documents do not contain sufficient information to answer this question accurately, "
        "as the text does not mention the name of the merchant's camel."
    )
    result = AnswerValidator.validate_answer(
        question="What is the name of the merchant's camel?",
        answer=llm_abstention,
        chunks=sample_context_chunk,
    )
    assert result["should_abstain"] is True
    assert result["validated_answer"] == ABSTENTION_MESSAGE


def test_numerical_grounding_validation(sample_context_chunk):
    context = sample_context_chunk[0].text  # "carrying silk and spices"

    # Answer contains hallucinated number "5000" not in context
    hallucinated_answer = "The merchant sold 5000 units of silk [Source: desert_tales.txt]."
    nums_ok, unsupported = AnswerValidator.validate_numerical_grounding(hallucinated_answer, context)
    assert nums_ok is False
    assert "5000" in unsupported

    # Answer without hallucinated numbers
    grounded_answer = "The merchant was carrying silk and spices across the desert [Source: desert_tales.txt]."
    nums_ok2, unsupported2 = AnswerValidator.validate_numerical_grounding(grounded_answer, context)
    assert nums_ok2 is True
    assert len(unsupported2) == 0
