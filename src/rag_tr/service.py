from dataclasses import dataclass
from pathlib import Path

import anthropic

from rag_tr.config import Settings
from rag_tr.generation.answerer import AnswerResult, generate_answer
from rag_tr.ingestion.chunker import chunk_text
from rag_tr.ingestion.loaders import load_document
from rag_tr.retrieval.embeddings import EmbeddingModel
from rag_tr.retrieval.hybrid import reciprocal_rank_fusion
from rag_tr.retrieval.keyword_search import BM25Index
from rag_tr.retrieval.vector_store import VectorStore


@dataclass
class IngestResult:
    ingested_files: list[str]
    failed_files: list[str]
    chunk_count: int


class RAGService:
    def __init__(
        self,
        settings: Settings,
        *,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        """Bagimliliklar test edilebilirlik icin enjekte edilebilir; verilmezse
        uretim davranisi aynen korunur (gercek model, kalici store, gercek client)."""
        self.settings = settings
        self.embedding_model = embedding_model or EmbeddingModel(settings.embedding_model_name)
        self.vector_store = vector_store or VectorStore(settings.chroma_persist_dir)
        self.bm25_index = bm25_index or BM25Index()
        self.client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def ingest_files(self, saved_paths: list[Path]) -> IngestResult:
        ingested: list[str] = []
        failed: list[str] = []
        total_chunks = 0

        for path in saved_paths:
            try:
                pages = load_document(path)
            except Exception:
                failed.append(path.name)
                continue

            self.vector_store.delete_by_source(path.name)
            self.bm25_index.remove_by_source(path.name)

            chunks = chunk_text(
                pages,
                source_file=path.name,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            )
            if chunks:
                embeddings = self.embedding_model.encode_passages([c.text for c in chunks])
                self.vector_store.add(chunks, embeddings)
                self.bm25_index.add(chunks)
                total_chunks += len(chunks)
            ingested.append(path.name)

        return IngestResult(ingested_files=ingested, failed_files=failed, chunk_count=total_chunks)

    def query(self, question: str, top_k: int) -> AnswerResult:
        if self.vector_store.count() == 0:
            return AnswerResult(answer="Dokümanlarda bu bilgi yok.", sources=[], used_chunk_ids=[])

        query_embedding = self.embedding_model.encode_query(question)
        vector_results = self.vector_store.query(query_embedding, self.settings.top_k_vector)
        keyword_results = self.bm25_index.query(question, self.settings.top_k_keyword)
        fused = reciprocal_rank_fusion(vector_results, keyword_results, k=self.settings.rrf_k)
        top_ids = [chunk_id for chunk_id, _ in fused[:top_k]]

        if not top_ids:
            return AnswerResult(answer="Dokümanlarda bu bilgi yok.", sources=[], used_chunk_ids=[])

        chunks = self.vector_store.get_chunks(top_ids)
        return generate_answer(question, chunks, self.client, self.settings.anthropic_model)
