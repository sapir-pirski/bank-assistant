from app.rag import MAX_MEMORY_TURNS, RAGService


def service_without_init() -> RAGService:
    return object.__new__(RAGService)


def test_append_history_keeps_last_memory_turns():
    service = service_without_init()
    history: list[dict[str, str]] = []

    for index in range(MAX_MEMORY_TURNS + 2):
        history = service._append_history(history, f"question {index}", f"answer {index}")

    assert len(history) == MAX_MEMORY_TURNS
    assert history[0]["question"] == "question 2"
    assert history[-1]["answer"] == "answer 7"


def test_retrieval_query_uses_memory_for_follow_up():
    service = service_without_init()
    history = [
        {
            "question": "What securities trading fees apply to the premium plan?",
            "answer": "premium plan securities fee answer.",
        }
    ]

    query = service._build_retrieval_query("And what about the standard plan?", history)

    assert "Previous user question" in query
    assert "premium plan" in query
    assert "Current retrieval question: And what about the standard plan?" in query


def test_retrieval_query_does_not_use_memory_for_standalone_question():
    service = service_without_init()
    history = [{"question": "Previous question", "answer": "Previous answer"}]

    query = service._build_retrieval_query("What are ATM withdrawal fees?", history)

    assert query == "What are ATM withdrawal fees?"
