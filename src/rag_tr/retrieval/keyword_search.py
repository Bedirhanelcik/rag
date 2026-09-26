import re
import threading

from rank_bm25 import BM25Okapi

from rag_tr.ingestion.chunker import Chunk
from rag_tr.retrieval.vector_store import make_chunk_id

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._lock = threading.Lock()

    def add(self, chunks: list[Chunk]) -> None:
        with self._lock:
            for chunk in chunks:
                self._chunk_ids.append(make_chunk_id(chunk.source_file, chunk.chunk_index))
                self._chunks.append(chunk)
            tokenized = [_tokenize(c.text) for c in self._chunks]
            self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def remove_by_source(self, source_file: str) -> None:
        with self._lock:
            kept_ids: list[str] = []
            kept_chunks: list[Chunk] = []
            for chunk_id, chunk in zip(self._chunk_ids, self._chunks):
                if chunk.source_file != source_file:
                    kept_ids.append(chunk_id)
                    kept_chunks.append(chunk)
            self._chunk_ids = kept_ids
            self._chunks = kept_chunks
            tokenized = [_tokenize(c.text) for c in self._chunks]
            self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def query(self, text: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda item: item[1], reverse=True)
        return [(chunk_id, score) for chunk_id, score in ranked[:top_k] if score > 0]
