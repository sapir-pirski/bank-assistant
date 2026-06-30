from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from app.config import Settings
from app.documents import DocumentChunk


class PolicyVectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))

    @property
    def collection_name(self) -> str:
        return self.settings.effective_collection_name

    def collection(self) -> Any:
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": self.settings.similarity_metric,
                "embedding_model": self.settings.embedding_model,
                "similarity_metric": self.settings.similarity_metric,
            },
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

    def count(self) -> int:
        try:
            return int(self.collection().count())
        except Exception:
            return 0

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]], batch_size: int = 128) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        collection = self.collection()
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]
            collection.add(
                ids=[chunk.id for chunk in batch_chunks],
                documents=[chunk.text for chunk in batch_chunks],
                metadatas=[chunk.metadata for chunk in batch_chunks],
                embeddings=batch_embeddings,
            )
        return len(chunks)

    def query(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        result = self.collection().query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            distance = float(distances[index]) if index < len(distances) else 1.0
            metadata = metadatas[index] if index < len(metadatas) else {}
            hits.append(
                {
                    "text": document,
                    "source": metadata.get("source", "unknown"),
                    "heading": metadata.get("heading", "Document"),
                    "chunk_index": metadata.get("chunk_index", index),
                    "distance": distance,
                    "relevance": max(0.0, 1.0 - distance),
                }
            )
        return hits
