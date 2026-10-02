"""Upload uc noktasinin HTTP sozlesmesi."""

from typing import Literal

from pydantic import BaseModel

UploadFileStatus = Literal["ingested", "failed"]


class UploadedFile(BaseModel):
    filename: str
    chunks_created: int
    status: UploadFileStatus


class UploadResponse(BaseModel):
    files: list[UploadedFile]
    total_chunks: int
    """Yukleme sonrasi korpusta bulunan toplam chunk sayisi -- yuklenen
    dokumanin artik sorgulanabilir oldugunu dogrulamak icin."""
    corpus_chunks: int
