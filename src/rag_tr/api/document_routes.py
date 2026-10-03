"""Korpus yonetimi uc noktalari.

Arayuzdeki dokuman alaninin ihtiyac duydugu uc islem: korpusta ne oldugunu
gormek, bir dokumani kaldirmak ve korpusu bosaltmak.

Yetkilendirme notu: listeleme HERKESE aciktir -- yalnizca dosya adlarini ve
chunk sayilarini gosterir, icerik dondurmez ve hicbir sey degistirmez. Silme ve
sifirlama ise YIKICIDIR ve korpus tum ziyaretciler arasinda paylasimli
oldugundan `UPLOAD_ENABLED` bayragina baglidir: yukleme kapaliyken bir
ziyaretcinin baskasinin dokumanini silebilmesi mumkun olmamali. Boylece
"yukleyebilen silebilir" kurali tek bir bayrakla ifade ediliyor.

`INGEST_API_TOKEN` buraya hic girmez: o token operator uc noktasi `/ingest`e
aittir ve tarayiciya asla ulasmaz.
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from rag_tr.api import routes
from rag_tr.api.document_schemas import (
    CorpusResetResponse,
    DocumentItem,
    DocumentListResponse,
    DocumentRemovedResponse,
)
from rag_tr.contracts import ErrorCode

router = APIRouter(prefix="/documents", tags=["documents"])


def _require_upload_enabled(settings) -> None:
    if not settings.upload_enabled:
        raise HTTPException(
            status_code=403,
            detail=routes._error(
                ErrorCode.UPLOAD_DISABLED,
                "Korpus bu ortamda salt okunur. Doküman ekleme ve silme yalnızca "
                "UPLOAD_ENABLED=true olan ortamlarda açıktır.",
            ),
        )


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
        # karar verir; kapali bir ortamda var olmayan bir eylemi sunmaz.
        can_modify=bool(service.settings.upload_enabled),
    )


@router.delete("/{source_file}", response_model=DocumentRemovedResponse)
async def remove_document(source_file: str, request: Request) -> DocumentRemovedResponse:
    service = request.app.state.service
    _require_upload_enabled(service.settings)

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
    _require_upload_enabled(service.settings)

    removed = await run_in_threadpool(service.reset_corpus)

    for path in routes.UPLOAD_DIR.glob("*"):
        if path.is_file():
            path.unlink()

    return CorpusResetResponse(
        chunks_removed=removed,
        total_chunks=service.vector_store.count(),
        keyword_index_size=service.bm25_index.size(),
    )
