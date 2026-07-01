from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _csv_from_env(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_collection_suffix(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:48] or "default"


@dataclass
class Settings:
    openai_api_key: str
    chat_model: str
    judge_model: str
    response_temperature: float | None
    embedding_model: str
    eval_embedding_models: list[str]
    eval_chat_models: list[str]
    similarity_metric: str
    eval_similarity_metrics: list[str]
    eval_top_ks: list[int]
    data_dir: Path
    chroma_path: Path
    chroma_collection: str
    retrieval_top_k: int
    max_retrieval_distance: float
    max_context_chunks: int
    max_sub_questions: int
    max_rag_retries: int
    quality_score_threshold: int
    auto_index_on_startup: bool

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(override=False)
        default_embedding = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4"),
            judge_model=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4.1"),
            response_temperature=_float_from_env("OPENAI_TEMPERATURE", 0.01),
            embedding_model=default_embedding,
            eval_embedding_models=_csv_from_env(
                "OPENAI_EVAL_EMBEDDING_MODELS",
                ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            ),
            eval_chat_models=_csv_from_env(
                "OPENAI_EVAL_CHAT_MODELS",
                ["gpt-5.5", "gpt-5.4", "gpt-4.1"],
            ),
            similarity_metric=os.getenv("CHROMA_SIMILARITY", "cosine"),
            eval_similarity_metrics=_csv_from_env(
                "OPENAI_EVAL_SIMILARITIES",
                ["cosine", "l2", "ip"],
            ),
            eval_top_ks=[int(item) for item in _csv_from_env("OPENAI_EVAL_TOP_KS", ["3", "5", "8"])],
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            chroma_path=Path(os.getenv("CHROMA_PATH", "chroma_data")),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "bank_policy"),
            retrieval_top_k=_int_from_env("RETRIEVAL_TOP_K", 3),
            max_retrieval_distance=_float_from_env("MAX_RETRIEVAL_DISTANCE", 0.85),
            max_context_chunks=_int_from_env("MAX_CONTEXT_CHUNKS", 8),
            max_sub_questions=_int_from_env("MAX_SUB_QUESTIONS", 4),
            max_rag_retries=_int_from_env("MAX_RAG_RETRIES", 2),
            quality_score_threshold=_int_from_env("QUALITY_SCORE_THRESHOLD", 4),
            auto_index_on_startup=_bool_from_env("AUTO_INDEX_ON_STARTUP", True),
        )

    @property
    def effective_collection_name(self) -> str:
        suffix = safe_collection_suffix(self.embedding_model)
        similarity = safe_collection_suffix(self.similarity_metric)
        return f"{self.chroma_collection}_{suffix}_{similarity}"

    def require_openai_key(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required. Copy .env.template to .env and set a key.")

    def with_overrides(self, **kwargs: object) -> "Settings":
        return replace(self, **kwargs)
