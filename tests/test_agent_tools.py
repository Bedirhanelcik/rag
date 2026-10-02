"""RagSearchTool adapter testleri.

Tool, RAGService.retrieve()'i sarar ve asla istisna firlatmaz: agent'in hata
uzerine muhakeme edebilmesi icin sonucu tiplenmis bir status olarak dondurur.
"""

import pytest

from rag_tr.agent.contracts import ToolStatus
from rag_tr.agent.tools import RagSearchTool
from rag_tr.contracts import Passage, RetrievalError, RetrievalResult, RetrievalStatus


def _passage(chunk_id="a.md::0", text="Ankara", rank=1, score=0.5) -> Passage:
    return Passage(
        chunk_id=chunk_id,
        source_file=chunk_id.split("::")[0],
        page_number=None,
        text=text,
        rank=rank,
        score=score,
    )


class _FakeService:
    """Yalnizca retrieve() yuzeyini taklit eder."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, int, str | None]] = []

    def retrieve(self, question: str, top_k: int = 5, source_file: str | None = None):
        self.calls.append((question, top_k, source_file))
        if self.error is not None:
            raise self.error
        return self.result


def test_tool_returns_ok_with_passages():
    service = _FakeService(
        RetrievalResult(status=RetrievalStatus.FOUND, passages=[_passage(), _passage("a.md::1", rank=2)])
    )
    tool = RagSearchTool(service)

    outcome = tool.search("Başkent neresi?", top_k=3)

    assert outcome.status is ToolStatus.OK
    assert len(outcome.passages) == 2
    assert outcome.error is None
    assert service.calls == [("Başkent neresi?", 3, None)]


def test_tool_returns_empty_when_retrieval_finds_nothing():
    service = _FakeService(RetrievalResult(status=RetrievalStatus.NO_RELEVANT_CONTEXT, passages=[]))
    tool = RagSearchTool(service)

    outcome = tool.search("Mars'ta hava?")

    assert outcome.status is ToolStatus.EMPTY
    assert outcome.passages == []
    assert outcome.error is None


def test_tool_converts_retrieval_error_into_failed_status_without_raising():
    service = _FakeService(error=RetrievalError("chroma down"))
    tool = RagSearchTool(service)

    outcome = tool.search("Başkent?")

    assert outcome.status is ToolStatus.FAILED
    assert outcome.passages == []
    assert "chroma down" in outcome.error


def test_tool_converts_unexpected_error_into_failed_status():
    service = _FakeService(error=RuntimeError("beklenmeyen"))
    tool = RagSearchTool(service)

    outcome = tool.search("Başkent?")

    assert outcome.status is ToolStatus.FAILED
    assert "beklenmeyen" in outcome.error


def test_tool_forwards_source_file_scope():
    service = _FakeService(RetrievalResult(status=RetrievalStatus.FOUND, passages=[_passage()]))
    tool = RagSearchTool(service)

    tool.search("Osmanlı?", top_k=2, source_file="tarih.md")

    assert service.calls == [("Osmanlı?", 2, "tarih.md")]


def test_tool_exposes_a_stable_name():
    assert RagSearchTool(_FakeService()).name == "rag_search"


@pytest.mark.parametrize("bad_top_k", [0, -1])
def test_tool_rejects_non_positive_top_k(bad_top_k):
    tool = RagSearchTool(_FakeService())

    with pytest.raises(ValueError):
        tool.search("Soru?", top_k=bad_top_k)
