from __future__ import annotations

from app.config import Settings
from app.documents import DocumentChunk, load_markdown_chunks
from app.openai_utils import build_openai_client
from app.vector_store import PolicyVectorStore


class PolicyIndexer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = build_openai_client(settings)
        self.store = PolicyVectorStore(settings)

    def ensure_index(self) -> int:
        existing_count = self.store.count()
        if existing_count > 0:
            return existing_count
        return self.reindex()

    def reindex(self) -> int:
        chunks = load_markdown_chunks(self.settings.data_dir)
        embeddings = self.embed_texts([chunk.text for chunk in chunks])
        self.store.reset()
        return self.store.add_chunks(chunks, embeddings)

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        embeddings: list[list[float]] = []
        clean_texts = [text.strip() for text in texts if text.strip()]
        for start in range(0, len(clean_texts), batch_size):
            batch = clean_texts[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=batch,
            )
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    def embed_query(self, question: str) -> list[float]:
        embedding, _usage = self.embed_query_with_usage(question)
        return embedding

    def embed_query_with_usage(self, question: str) -> tuple[list[float], dict[str, int | None]]:
        response = self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=question.strip(),
        )
        return response.data[0].embedding, _embedding_usage(response)


def _embedding_usage(response: object) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if not usage:
        return {}
    input_tokens = getattr(usage, "prompt_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "input_tokens", None)
    return {
        "input_tokens": input_tokens,
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def index_documents(settings: Settings) -> int:
    return PolicyIndexer(settings).reindex()
