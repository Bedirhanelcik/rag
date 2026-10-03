"""Calisan servise HTTP uzerinden dokuman ingest eder.

Neden HTTP: ingestion, calisan FastAPI process'i uzerinden yapilmali. Ayri bir
process acip Chroma'ya yazmak yalnizca vektor deposunu guncellerdi; sunucunun
bellekteki BM25 index'i yeniden baslatmaya kadar eski kalir ve hibrit arama
sessizce yalniz-vektor moduna duserdi. `/ingest` ikisini birlikte gunceller.

Render Free'de kalici disk olmadigi icin bu script'in her deploy sonrasi bir kez
calistirilmasi gerekir.

Kullanim:
    export RAG_API_URL=https://<servis>.onrender.com
    export INGEST_API_TOKEN=<panelden alinan deger>
    uv run --no-editable python scripts/ingest.py                 # data/sample/*
    uv run --no-editable python scripts/ingest.py belge.pdf not.md

Token ve URL yalnizca ortam degiskeninden okunur; hicbir yere yazdirilmaz.
"""

import os
import sys
from pathlib import Path

import requests

SUPPORTED = {".pdf", ".txt", ".md"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "sample"
TIMEOUT_SECONDS = 300


def _collect(arguments: list[str]) -> list[Path]:
    if arguments:
        paths = [Path(argument) for argument in arguments]
    else:
        paths = sorted(p for p in DEFAULT_SOURCE.glob("*") if p.suffix.lower() in SUPPORTED)

    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"Dosya bulunamadı: {', '.join(str(p) for p in missing)}")

    unsupported = [p for p in paths if p.suffix.lower() not in SUPPORTED]
    if unsupported:
        raise SystemExit(
            f"Desteklenmeyen tür: {', '.join(p.name for p in unsupported)} "
            f"(izin verilen: {', '.join(sorted(SUPPORTED))})"
        )
    if not paths:
        raise SystemExit(f"Gönderilecek dosya yok ({DEFAULT_SOURCE} boş).")
    return paths


def main() -> int:
    base_url = (os.environ.get("RAG_API_URL") or "").strip().rstrip("/")
    token = (os.environ.get("INGEST_API_TOKEN") or "").strip()

    if not base_url:
        raise SystemExit("RAG_API_URL tanımlı değil (örn. https://<servis>.onrender.com).")
    if not token:
        raise SystemExit("INGEST_API_TOKEN tanımlı değil; /ingest token olmadan 503 döner.")

    paths = _collect(sys.argv[1:])
    print(f"hedef   : {base_url}/ingest")
    print(f"dosyalar: {', '.join(p.name for p in paths)}")

    handles = [("files", (p.name, p.open("rb"))) for p in paths]
    try:
        response = requests.post(
            f"{base_url}/ingest",
            files=handles,
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUT_SECONDS,
        )
    finally:
        for _, (_, handle) in handles:
            handle.close()

    if response.status_code != 200:
        # Token hicbir zaman yazdirilmaz; yalnizca sunucunun hata kodu gosterilir.
        detail = {}
        try:
            detail = response.json().get("detail", {})
        except ValueError:
            pass
        print(f"BAŞARISIZ  HTTP {response.status_code}  kod={detail.get('code', '?')}")
        print(f"  {detail.get('message', response.text[:200])}")
        return 1

    body = response.json()
    print(f"ingest   : {body['ingested_files']}")
    if body["failed_files"]:
        print(f"başarısız: {body['failed_files']}")
    print(f"chunk    : {body['chunk_count']}")

    health = requests.get(f"{base_url}/health", timeout=30).json()
    print(f"korpus   : {health['chunk_count']} chunk / BM25 {health['keyword_index_size']}")
    return 0 if body["chunk_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
