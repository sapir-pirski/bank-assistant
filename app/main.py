from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import time
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import Settings
from app.indexer import index_documents
from app.rag import RAGService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

settings = Settings.from_env()
rag_service = RAGService(settings)

app = FastAPI(
    title="ONE ZERO Policy Chatbot",
    description="RAG chatbot over provided ONE ZERO bank policy documents.",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)
    session_id: str | None = Field(default=None, min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_.:-]+$")


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    guardrail_reason: str | None = None
    session_id: str
    classification: str | None = None
    classification_reason: str | None = None
    sub_questions: list[str] = Field(default_factory=list)
    quality_score: dict[str, Any] = Field(default_factory=dict)
    output_validation: dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)
    model: str
    embedding_model: str
    similarity_metric: str
    collection: str
    generation_metadata: dict[str, Any] = Field(default_factory=dict)


@app.on_event("startup")
def startup_index() -> None:
    logger.info(
        "Starting policy chatbot model=%s judge_model=%s embedding=%s similarity=%s top_k=%s collection=%s temperature=%s",
        settings.chat_model,
        settings.judge_model,
        settings.embedding_model,
        settings.similarity_metric,
        settings.retrieval_top_k,
        settings.effective_collection_name,
        settings.response_temperature,
    )
    if not settings.auto_index_on_startup:
        logger.info("Startup indexing disabled")
        return
    try:
        count = rag_service.indexer.ensure_index()
        logger.info("Policy vector index ready with %s chunks", count)
    except Exception:
        logger.exception("Policy vector index was not created during startup")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "collection": settings.effective_collection_name,
        "indexed_chunks": rag_service.store.count(),
    }


@app.post("/api/index")
def rebuild_index() -> dict[str, Any]:
    try:
        started = time.perf_counter()
        count = index_documents(settings)
    except Exception as exc:
        logger.exception("Manual index rebuild failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    logger.info(
        "Manual index rebuild completed chunks=%s collection=%s duration_ms=%s",
        count,
        settings.effective_collection_name,
        round((time.perf_counter() - started) * 1000),
    )
    return {
        "indexed_chunks": count,
        "collection": settings.effective_collection_name,
        "embedding_model": settings.embedding_model,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict[str, Any]:
    session_id = request.session_id or "default"
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    message_hash = hashlib.sha256(request.message.encode("utf-8")).hexdigest()[:12]
    logger.info(
        "Chat request received request_id=%s session_id=%s message_len=%s message_hash=%s requested_top_k=%s",
        request_id,
        session_id,
        len(request.message),
        message_hash,
        request.top_k,
    )
    try:
        result = rag_service.answer(request.message, request.top_k, session_id=session_id, request_id=request_id)
    except Exception as exc:
        logger.exception("Chat request failed request_id=%s session_id=%s message_hash=%s", request_id, session_id, message_hash)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    latency_ms = round((time.perf_counter() - started) * 1000)
    result.setdefault("trace", {})["latency_ms"] = latency_ms
    logger.info(
        "Chat request completed request_id=%s session_id=%s latency_ms=%s sources=%s guardrail=%s model=%s embedding=%s similarity=%s",
        request_id,
        session_id,
        latency_ms,
        len(result.get("sources", [])),
        result.get("guardrail_reason") or "none",
        result.get("model"),
        result.get("embedding_model"),
        result.get("similarity_metric"),
    )
    logger.info(
        "Chat quality request_id=%s session_id=%s classification=%s score=%s retries=%s validation=%s embedding_calls=%s total_tokens=%s",
        request_id,
        session_id,
        result.get("classification") or "unknown",
        result.get("quality_score", {}).get("overall_score"),
        result.get("retry_count", 0),
        result.get("output_validation", {}).get("valid"),
        result.get("retrieval_metadata", {}).get("embedding_calls", 0),
        result.get("trace", {}).get("token_usage", {}).get("all_total_tokens", 0),
    )
    logger.info(
        "Chat trace request_id=%s usage_by_step=%s token_usage=%s",
        request_id,
        result.get("trace", {}).get("usage_by_step", {}),
        result.get("trace", {}).get("token_usage", {}),
    )
    return result
