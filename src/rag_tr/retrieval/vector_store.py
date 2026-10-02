import numpy as np
import chromadb

from rag_tr.ingestion.chunker import Chunk


def make_chunk_id(source_file: str, chunk_index: int) -> str:
    return f"{source_file}::{chunk_index}"


class VectorStore:
    def __init__(self, persist_dir: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection("chunks")

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        ids = [make_chunk_id(c.source_file, c.chunk_index) for c in chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "page_number": c.page_number if c.page_number is not None else -1,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        documents = [c.text for c in chunks]
        self._collection.upsert(
            ids=ids,
            embeddings=np.asarray(embeddings).tolist(),
            metadatas=metadatas,
            documents=documents,
        )

    def delete_by_source(self, source_file: str) -> None:
        self._collection.delete(where={"source_file": source_file})

    def query(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(query_embedding).tolist()],
            n_results=min(top_k, count),
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return list(zip(ids, distances))

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        result = self._collection.get(ids=chunk_ids)
        by_id = {
            id_: Chunk(
                text=doc,
                source_file=meta["source_file"],
                page_number=None if meta["page_number"] == -1 else meta["page_number"],
                chunk_index=meta["chunk_index"],
            )
            for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
        }
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def count(self) -> int:
        return self._collection.count()

    def all_chunks(self) -> list[Chunk]:
        """Kalici koleksiyondaki tum chunk'lari dondurur. BM25 index'i yeniden
        kurmak icin kullanilir: metin ve metadata zaten burada sakli oldugundan
        ayri bir kalicilik katmanina gerek kalmiyor."""
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            Chunk(
                text=doc,
                source_file=meta["source_file"],
                page_number=None if meta["page_number"] == -1 else meta["page_number"],
                chunk_index=meta["chunk_index"],
            )
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]
