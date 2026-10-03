"""Korpus yonetimi sozlesmesi.

Yalnizca dosya adi ve sayilar disa acilir; chunk metinleri, embedding'ler ve
depolama ayrintilari bu yuzeyin disinda kalir.
"""

from pydantic import BaseModel


class DocumentItem(BaseModel):
    source_file: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]
    total_chunks: int
    keyword_index_size: int
    #: Arayuzun silme/sifirlama eylemlerini gosterip gostermeyecegi.
    can_modify: bool


class DocumentRemovedResponse(BaseModel):
    source_file: str
    chunks_removed: int
    total_chunks: int
    keyword_index_size: int


class CorpusResetResponse(BaseModel):
    chunks_removed: int
    total_chunks: int
    keyword_index_size: int
