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
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._lock = threading.Lock()

    def _rebuild(self) -> None:
        """Lock tutulurken cagrilir."""
        self._bm25 = BM25Okapi(list(self._tokenized)) if self._tokenized else None

    def add(self, chunks: list[Chunk]) -> None:
        with self._lock:
            for chunk in chunks:
                self._chunk_ids.append(make_chunk_id(chunk.source_file, chunk.chunk_index))
                self._chunks.append(chunk)
                self._tokenized.append(_tokenize(chunk.text))
            self._rebuild()

    def remove_by_source(self, source_file: str) -> None:
        with self._lock:
            kept_ids: list[str] = []
            kept_chunks: list[Chunk] = []
            kept_tokens: list[list[str]] = []
            for chunk_id, chunk, tokens in zip(self._chunk_ids, self._chunks, self._tokenized):
                if chunk.source_file != source_file:
                    kept_ids.append(chunk_id)
                    kept_chunks.append(chunk)
                    kept_tokens.append(tokens)
            self._chunk_ids = kept_ids
            self._chunks = kept_chunks
            self._tokenized = kept_tokens
            self._rebuild()

    def size(self) -> int:
        with self._lock:
            return len(self._chunk_ids)

    def query(self, text: str, top_k: int) -> list[tuple[str, float]]:
        tokens = _tokenize(text)
        query_tokens = set(tokens)
        if not query_tokens:
            return []

        # add()/remove_by_source() _chunk_ids ile _bm25'i ayri adimlarda guncelliyor;
        # ayni lock olmadan okuma iki farkli nesli eslestirebilir.
        with self._lock:
            if self._bm25 is None:
                return []
            scores = self._bm25.get_scores(tokens)
            # Skor esigi yerine token ortusmesi: BM25Okapi, bir terim corpus'taki tum
            # dokumanlarda geciyorsa negatif IDF uretir ve "score > 0" filtresi o durumda
            # ilgili chunk'larin tamamini dusurur. RRF yalnizca rank kullandigi icin
            # mutlak skor degeri degil, sadece siralama onemli.
            candidates = [
                (chunk_id, float(score))
                for chunk_id, score, chunk_tokens in zip(self._chunk_ids, scores, self._tokenized)
                if query_tokens & set(chunk_tokens)
            ]

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:top_k]
