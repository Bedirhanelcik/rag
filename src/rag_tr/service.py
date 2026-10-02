from dataclasses import dataclass
from pathlib import Path

import anthropic

from rag_tr.config import Settings
from rag_tr.contracts import (
    GenerationError,
    Passage,
    RetrievalError,
    RetrievalResult,
    RetrievalStatus,
)
from rag_tr.generation.answerer import AnswerResult, generate_answer, no_context_result
from rag_tr.ingestion.chunker import Chunk, chunk_text
from rag_tr.ingestion.loaders import load_document
from rag_tr.retrieval.embeddings import EmbeddingModel
from rag_tr.retrieval.hybrid import reciprocal_rank_fusion
from rag_tr.retrieval.keyword_search import BM25Index
from rag_tr.retrieval.vector_store import VectorStore, make_chunk_id


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
        self.embedding_model = embedding_model or EmbeddingModel(
            settings.embedding_model_name,
            api_key=settings.gemini_api_key,
            dimensions=settings.embedding_dimensions,
        )
        self.vector_store = vector_store or VectorStore(settings.chroma_persist_dir)
        self.bm25_index = bm25_index or BM25Index()
        self.client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.rebuild_keyword_index()

    def rebuild_keyword_index(self) -> int:
        """BM25 index'i kalici vektor deposundan yeniden kurar.

        BM25 yalnizca bellekte yasiyor; ChromaDB ise chunk metinlerini ve
        metadata'sini diskte tutuyor. Servis her ayaga kalktiginda index'i o
        kalici veriden uretmek, hem yeniden baslatma sonrasi keyword aramasinin
        sessizce devre disi kalmasini onler hem de iki depo arasinda tutarsizlik
        birakmaz: tek kaynak-of-truth vektor deposudur."""
        chunks = self.vector_store.all_chunks()
        self.bm25_index.replace_all(chunks)
        return len(chunks)

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

    def _retrieve_chunks(
        self, question: str, top_k: int, source_file: str | None = None
    ) -> list[tuple[Chunk, float]]:
        """Hibrit retrieval'in tek implementasyonu: embedding -> vektor arama +
        BM25 -> RRF. Claude cagrilmaz. Hem retrieve() hem query() bunu kullanir,
        boylece iki yolun sonuclari hicbir zaman ayrismaz."""
        try:
            if self.vector_store.count() == 0:
                return []

            query_embedding = self.embedding_model.encode_query(question)
            vector_results = self.vector_store.query(
                query_embedding, self.settings.top_k_vector, source_file=source_file
            )
            keyword_results = self.bm25_index.query(
                question, self.settings.top_k_keyword, source_file=source_file
            )
            fused = reciprocal_rank_fusion(vector_results, keyword_results, k=self.settings.rrf_k)
            top = fused[:top_k]
            if not top:
                return []

            by_id = {
                make_chunk_id(c.source_file, c.chunk_index): c
                for c in self.vector_store.get_chunks([chunk_id for chunk_id, _ in top])
            }
        except Exception as exc:
            raise RetrievalError(str(exc)) from exc

        return [(by_id[chunk_id], score) for chunk_id, score in top if chunk_id in by_id]

    def retrieve(
        self, question: str, top_k: int = 5, source_file: str | None = None
    ) -> RetrievalResult:
        """Generation yapmadan yalnizca ilgili pasajlari dondurur (agent tool'u).

        `source_file` verilirse arama yalnizca o dosyanin chunk'lari uzerinde
        yapilir. Claude/Anthropic API cagrilmaz."""
        scored = self._retrieve_chunks(question, top_k, source_file)
        if not scored:
            return RetrievalResult(status=RetrievalStatus.NO_RELEVANT_CONTEXT, passages=[])

        passages = [
            Passage(
                chunk_id=make_chunk_id(chunk.source_file, chunk.chunk_index),
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                text=chunk.text,
                rank=rank,
                score=float(score),
            )
            for rank, (chunk, score) in enumerate(scored, start=1)
        ]
        return RetrievalResult(status=RetrievalStatus.FOUND, passages=passages)

    def query(self, question: str, top_k: int) -> AnswerResult:
        """Retrieval + Claude generation. Retrieval ve generation hatalarini ayri
        tiplere sararak yukari tasir; boylece API katmani (ve onu tool olarak
        cagiran agent) hangi asamanin basarisiz oldugunu serbest metne bakmadan
        ayirt edebilir."""
        scored = self._retrieve_chunks(question, top_k)
        if not scored:
            return no_context_result()

        chunks = [chunk for chunk, _score in scored]
        try:
            return generate_answer(question, chunks, self.client, self.settings.anthropic_model)
        except Exception as exc:
            raise GenerationError(str(exc)) from exc
