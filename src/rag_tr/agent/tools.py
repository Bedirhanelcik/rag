"""Agent'in kullanabilecegi tool'lar. Ilk surumde tek tool: RAG aramasi."""

from typing import Protocol

from rag_tr.agent.contracts import SearchOutcome, ToolStatus
from rag_tr.contracts import RetrievalStatus


class SearchTool(Protocol):
    name: str

    def search(
        self, query: str, top_k: int = 5, source_file: str | None = None
    ) -> SearchOutcome: ...


class RagSearchTool:
    """RAGService.retrieve() uzerine ince bir adapter.

    Iki is yapar: hatalari tiplenmis bir ToolStatus'a cevirir (agent'in hata
    uzerine muhakeme edebilmesi icin istisna firlatmaz) ve VectorStore /
    BM25Index / EmbeddingModel gibi ic detaylari agent'tan gizler -- agent
    yalnizca rag_tr.contracts tiplerini gorur."""

    name = "rag_search"

    def __init__(self, service) -> None:
        self._service = service

    def search(self, query: str, top_k: int = 5, source_file: str | None = None) -> SearchOutcome:
        if top_k < 1:
            raise ValueError("top_k en az 1 olmali")
        try:
            result = self._service.retrieve(query, top_k=top_k, source_file=source_file)
        except Exception as exc:
            # Retrieval hatasi agent'i cokertmemeli; karar verebilecegi bir
            # sonuca donusturulur.
            return SearchOutcome(status=ToolStatus.FAILED, passages=[], error=str(exc))

        if result.status is RetrievalStatus.NO_RELEVANT_CONTEXT or not result.passages:
            return SearchOutcome(status=ToolStatus.EMPTY, passages=[])
        return SearchOutcome(status=ToolStatus.OK, passages=list(result.passages))
