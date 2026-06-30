from app.rag import RAGService


def service_without_init() -> RAGService:
    return object.__new__(RAGService)


def test_format_sources_numbers_and_shortens_previews():
    service = service_without_init()
    documents = [
        {
            "source": "data/cards.md",
            "heading": "Cards > ATM",
            "retrieved_for": "ATM fees?",
            "distance": 0.12345,
            "relevance": 0.87654,
            "text": "A" * 400,
        },
        {
            "source": "data/securities.md",
            "heading": "Securities > Fees",
            "distance": 0.2,
            "relevance": 0.8,
            "text": "Fee policy text.",
        },
    ]

    sources = service._format_sources(documents)

    assert [source["id"] for source in sources] == [1, 2]
    assert sources[0]["source"] == "data/cards.md"
    assert sources[0]["heading"] == "Cards > ATM"
    assert sources[0]["retrieved_for"] == "ATM fees?"
    assert sources[0]["distance"] == 0.1235
    assert sources[0]["relevance"] == 0.8765
    assert len(sources[0]["preview"]) == 260
    assert sources[1]["preview"] == "Fee policy text."
