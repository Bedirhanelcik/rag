import threading
import time

from rag_tr.ingestion.chunker import Chunk
from rag_tr.retrieval import keyword_search
from rag_tr.retrieval.keyword_search import BM25Index


def _chunks(source_file: str, word: str, count: int) -> list[Chunk]:
    return [
        Chunk(
            text=f"{word} hakkinda metin numara {i}",
            source_file=source_file,
            page_number=None,
            chunk_index=i,
        )
        for i in range(count)
    ]


def test_query_acquires_the_index_lock():
    index = BM25Index()
    index.add(_chunks("a.txt", "alfa", 2))

    class _TrackingLock:
        def __init__(self, inner):
            self._inner = inner
            self.enter_count = 0

        def __enter__(self):
            self.enter_count += 1
            return self._inner.__enter__()

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

    tracking = _TrackingLock(index._lock)
    index._lock = tracking

    index.query("alfa", 5)

    assert tracking.enter_count == 1, "query() lock almiyor"


def test_query_does_not_observe_half_updated_index(monkeypatch):
    """remove_by_source() _chunk_ids'i degistirip _bm25'i yeniden kurarken araya giren
    bir query, iki ayri nesilden gelen (chunk_ids, scores) ciftini eslestirmemeli."""
    index = BM25Index()
    index.add(_chunks("a.txt", "alfa", 5))
    index.add(_chunks("b.txt", "beta", 5))

    real_bm25 = keyword_search.BM25Okapi
    entered = threading.Event()
    release = threading.Event()

    class _SlowBM25(real_bm25):
        def __init__(self, corpus):
            entered.set()
            release.wait(10)
            super().__init__(corpus)

    monkeypatch.setattr(keyword_search, "BM25Okapi", _SlowBM25)

    remover = threading.Thread(target=lambda: index.remove_by_source("a.txt"))
    remover.start()
    assert entered.wait(10), "remove_by_source yeniden kurma asamasina gelmedi"

    results: dict[str, list[tuple[str, float]]] = {}
    querier = threading.Thread(target=lambda: results.__setitem__("hits", index.query("beta", 10)))
    querier.start()
    time.sleep(0.1)

    release.set()
    remover.join(10)
    querier.join(10)

    hits = results["hits"]
    assert hits, "yarim guncellenmis index'ten okuma yapildi: beta sorgusu bos dondu"
    for chunk_id, _score in hits:
        assert chunk_id.startswith("b.txt::"), f"tutarsiz eslestirme: {chunk_id}"


def test_remove_by_source_then_query_returns_only_remaining_source():
    index = BM25Index()
    index.add(_chunks("a.txt", "alfa", 3))
    index.add(_chunks("b.txt", "beta", 3))

    index.remove_by_source("a.txt")

    assert index.query("alfa", 10) == []
    hits = index.query("beta", 10)
    assert hits
    assert all(chunk_id.startswith("b.txt::") for chunk_id, _ in hits)


def test_term_present_in_every_document_still_returns_hits():
    """BM25Okapi, tum dokumanlarda gecen bir terim icin negatif IDF uretir; eski
    "score > 0" filtresi bu durumda butun keyword listesini dusuruyordu."""
    index = BM25Index()
    index.add(_chunks("a.txt", "beta", 3))

    hits = index.query("beta", 10)

    assert len(hits) == 3


def test_query_ignores_chunks_without_any_query_token():
    index = BM25Index()
    index.add(_chunks("a.txt", "alfa", 2))
    index.add(_chunks("b.txt", "beta", 2))

    hits = index.query("alfa", 10)

    assert all(chunk_id.startswith("a.txt::") for chunk_id, _ in hits)
    assert len(hits) == 2


def test_size_reports_corpus_length():
    index = BM25Index()
    assert index.size() == 0
    index.add(_chunks("a.txt", "alfa", 3))
    assert index.size() == 3
    index.remove_by_source("a.txt")
    assert index.size() == 0
