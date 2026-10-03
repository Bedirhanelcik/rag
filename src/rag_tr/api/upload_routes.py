"""POST /upload -- web arayuzunun kullandigi dokuman yukleme uc noktasi.

Mevcut `/ingest` davranisi degistirilmez ve ingestion mantigi yeniden yazilmaz:
dosya adi sanitizasyonu (`_safe_filename`) ve yukleme dizini mevcut route
modulunden, chunk'lama/embedding/indeksleme ise `RAGService.ingest_files`'tan
oldugu gibi kullanilir. Bu uc noktanin `/ingest`e ekledigi uc sey var:

  1. `UPLOAD_ENABLED` bayragi -- varsayilan KAPALI. Mevcut Chroma koleksiyonu
     tum ziyaretciler arasinda paylasimli oldugundan (kullanici izolasyonu bu
     fazin kapsaminda degil) dagitimda upload acik birakilmamali.
  2. Dosya boyutu siniri.
  3. Dosya basina sonuc: filename + chunks_created + status.
"""

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

# Guvenlik mantigi tek yerde kalsin diye sanitizer kopyalanmiyor. Modul
# uzerinden referans veriliyor (isimle degil): yukleme dizini tek bir yerde
# tanimli kaliyor ve testte tek noktadan degistirilebiliyor.
from rag_tr.api import routes
from rag_tr.api.upload_schemas import UploadedFile, UploadResponse
from rag_tr.contracts import ErrorCode

router = APIRouter(tags=["upload"])


def _too_large(name: str, limit: int) -> HTTPException:
    return HTTPException(
        status_code=413,
        detail=routes._error(
            ErrorCode.FILE_TOO_LARGE,
            f"{name} boyut sınırını aşıyor (en fazla {limit} bayt).",
        ),
    )


@router.post("/upload", response_model=UploadResponse)
async def upload(
    request: Request, files: list[UploadFile] = File(...)
) -> UploadResponse:
    service = request.app.state.service
    settings = service.settings

    if not settings.upload_enabled:
        raise HTTPException(
            status_code=403,
            detail=routes._error(
                ErrorCode.UPLOAD_DISABLED,
                "Doküman yükleme bu ortamda kapalı. Korpus ziyaretçiler arasında "
                "paylaşımlı olduğu için yükleme yalnızca yerel geliştirmede açılır.",
            ),
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail=routes._error(ErrorCode.INVALID_FILENAME, "Yüklenecek dosya verilmedi."),
        )

    limit = settings.upload_max_bytes
    routes.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    results: list[UploadedFile] = []
    total_chunks = 0

    for file in files:
        # Ad sanitizasyonu ve uzanti kontrolu mevcut /ingest ile ayni fonksiyon:
        # path bilesenleri dusurulur, desteklenmeyen uzanti 400 verir.
        name = routes._safe_filename(file.filename)

        # Istemcinin bildirdigi boyut varsa once ona bakilir; boylece cok buyuk
        # bir govde hic belege alinmaz.
        declared = getattr(file, "size", None)
        if isinstance(declared, int) and declared > limit:
            raise _too_large(name, limit)

        data = await file.read()
        if len(data) > limit:
            raise _too_large(name, limit)

        dest = (routes.UPLOAD_DIR / name).resolve()
        if routes.UPLOAD_DIR.resolve() not in dest.parents:
            raise HTTPException(
                status_code=400,
                detail=routes._error(ErrorCode.INVALID_FILENAME, "Geçersiz dosya adı."),
            )
        dest.write_bytes(data)

        # Dosya basina chunk sayisi icin ayni pipeline tek dosyayla cagrilir;
        # ingest_files zaten dosyalar uzerinde donuyor, dolayisiyla fazladan is
        # yapilmiyor ve servis kodu degismiyor.
        result = await run_in_threadpool(service.ingest_files, [dest])
        ingested = name in result.ingested_files
        total_chunks += result.chunk_count
        results.append(
            UploadedFile(
                filename=name,
                chunks_created=result.chunk_count,
                status="ingested" if ingested else "failed",
            )
        )

    return UploadResponse(
        files=results,
        total_chunks=total_chunks,
        corpus_chunks=service.vector_store.count(),
    )
