"""Korpus yonetimi uc noktalari.

Arayuzdeki dokuman alaninin ihtiyac duydugu uc islem: korpusta ne oldugunu
gormek, bir dokumani kaldirmak ve korpusu bosaltmak.

Yetkilendirme notu: listeleme HERKESE aciktir -- yalnizca dosya adlarini ve
chunk sayilarini gosterir, icerik dondurmez ve hicbir sey degistirmez. Silme ve
sifirlama ise korpusu DEGISTIRIR ve `INGEST_API_TOKEN` ile korunur; kontrol
`api.auth` icinde, `/upload` ve `/ingest` ile ayni yerde.

Token tarayiciya asla ulasmaz: web arayuzu bu uclari kendi sunucusu uzerinden
cagirir ve basligi orada ekler.
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from rag_tr.api import routes
from rag_tr.api.auth import has_write_access, require_write_token
from rag_tr.api.document_schemas import (
    CorpusResetResponse,
    DocumentItem,
    DocumentListResponse,
    DocumentRemovedResponse,
)
from rag_tr.contracts import ErrorCode

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
async def list_documents(request: Request) -> DocumentListResponse:
    service = request.app.state.service
    documents = await run_in_threadpool(service.list_documents)

    return DocumentListResponse(
        documents=[
            DocumentItem(source_file=item.source_file, chunk_count=item.chunk_count)
            for item in documents
        ],
        total_chunks=service.vector_store.count(),
        keyword_index_size=service.bm25_index.size(),
        # Arayuz, silme dugmelerini gosterip gostermeyecegine buna bakarak
        # karar verir. Olcut sunucunun token'i olmasi DEGIL, bu istegin gecerli
        # bir token tasimasi: aksi halde anonim bir tarayiciya silme dugmesi
        # gosterilir ve dugme 401 ile donerdi.
        can_modify=has_write_access(request, service.settings),
    )


@router.delete("/{source_file}", response_model=DocumentRemovedResponse)
async def remove_document(source_file: str, request: Request) -> DocumentRemovedResponse:
    service = request.app.state.service
    require_write_token(request, service.settings)

    # Ad sanitizasyonu yukleme yoluyla ayni fonksiyondan geliyor: yol bileseni
    # tasiyan bir ad burada da kabul edilmez.
    name = routes._safe_filename(source_file)

    removed = await run_in_threadpool(service.remove_document, name)
    if removed == 0:
        raise HTTPException(
            status_code=404,
            detail=routes._error(
                ErrorCode.DOCUMENT_NOT_FOUND,
                f"Korpusta {name} adlı bir doküman yok.",
            ),
        )

    # Diskteki kopya da kaldirilir, yoksa ayni ad yeniden yuklendiginde eski
    # dosya sessizce yerinde kalirdi.
    upload_copy = (routes.UPLOAD_DIR / name).resolve()
    if routes.UPLOAD_DIR.resolve() in upload_copy.parents and upload_copy.is_file():
        upload_copy.unlink()

    return DocumentRemovedResponse(
        source_file=name,
        chunks_removed=removed,
        total_chunks=service.vector_store.count(),
        keyword_index_size=service.bm25_index.size(),
    )


@router.post("/reset", response_model=CorpusResetResponse)
async def reset_corpus(request: Request) -> CorpusResetResponse:
    service = request.app.state.service
    require_write_token(request, service.settings)

    removed = await run_in_threadpool(service.reset_corpus)

    for path in routes.UPLOAD_DIR.glob("*"):
        if path.is_file():
            path.unlink()

    return CorpusResetResponse(
        chunks_removed=removed,
        total_chunks=service.vector_store.count(),
        keyword_index_size=service.bm25_index.size(),
    )
