from app.rag import RAGService


def service_without_init() -> RAGService:
    return object.__new__(RAGService)


def test_smalltalk_classification_skips_retrieval():
    service = service_without_init()

    result = service._smalltalk_classification("Hi")

    assert result is not None
    assert result["classification"] == "smalltalk"


def test_heuristic_classifies_bank_topic_outside_guides():
    service = service_without_init()

    result = service._heuristic_classification("What mortgage rates does the bank offer?")

    assert result["classification"] == "bank_other_topic"


def test_heuristic_classifies_simple_in_scope_question():
    service = service_without_init()

    result = service._heuristic_classification("What are card fees abroad?")

    assert result["classification"] == "in_scope_simple"


def test_heuristic_classifies_complex_in_scope_question():
    service = service_without_init()

    result = service._heuristic_classification("What are card fees abroad and premium plan securities fees?")

    assert result["classification"] == "in_scope_complex"
