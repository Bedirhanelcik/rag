from pathlib import Path

import anthropic
from fastapi import APIRouter, HTTPException, Request, UploadFile

from rag_tr.api.schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
)

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, files: list[UploadFile]) -> IngestResponse:
    service = request.app.state.service
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for file in files:
        dest = UPLOAD_DIR / file.filename
        dest.write_bytes(await file.read())
        saved_paths.append(dest)

    result = service.ingest_files(saved_paths)
    return IngestResponse(
        ingested_files=result.ingested_files,
        failed_files=result.failed_files,
        chunk_count=result.chunk_count,
    )


@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    service = request.app.state.service
    top_k = payload.top_k or service.settings.top_k_final
    try:
        result = service.query(payload.question, top_k)
    except anthropic.APIError as exc:
        raise HTTPException(status_code=502, detail=f"Claude API hatası: {exc}") from exc
    return QueryResponse(
        answer=result.answer,
        sources=[SourceItem(**source) for source in result.sources],
        used_chunk_ids=result.used_chunk_ids,
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    service = request.app.state.service
    return HealthResponse(
        status="ok",
        embedding_model=service.settings.embedding_model_name,
        chunk_count=service.vector_store.count(),
    )
