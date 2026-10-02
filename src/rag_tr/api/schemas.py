from pydantic import BaseModel, Field

from rag_tr.contracts import QueryStatus


class IngestResponse(BaseModel):
    ingested_files: list[str]
    failed_files: list[str]
    chunk_count: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    # ge=1: top_k=0 eskiden falsy oldugu icin sessizce varsayilana donuyordu;
    # agent icin sessiz duzeltme yerine acik bir dogrulama hatasi daha guvenli.
    top_k: int | None = Field(default=None, ge=1)


class SourceItem(BaseModel):
    index: int
    source_file: str
    page_number: int | None
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    used_chunk_ids: list[str]
    status: QueryStatus


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    chunk_count: int
    keyword_index_size: int


class ErrorDetail(BaseModel):
    """Hata yanitlarindaki `detail` govdesi."""

    code: str
    message: str
