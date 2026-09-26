from pydantic import BaseModel


class IngestResponse(BaseModel):
    ingested_files: list[str]
    failed_files: list[str]
    chunk_count: int


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceItem(BaseModel):
    index: int
    source_file: str
    page_number: int | None
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    used_chunk_ids: list[str]


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    chunk_count: int
    keyword_index_size: int
