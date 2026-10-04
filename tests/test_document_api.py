"""Korpus yonetimi uc noktalari: /documents.

Iki katman ayri ayri sinanir:

* Servis katmani GERCEK bir RAGService ile calisir (fake embedding modeli ve
  bellek ici bir vektor deposu kullanilarak), cunku bu testlerin asil amaci
  "iki indeks senkron kaliyor mu" sorusunu cevaplamak. Yalnizca Chroma'dan
  silip BM25'i eski haliyle birakan bir hata, HTTP seviyesinde gorunmezdi.
* HTTP katmani fake servisle sinanir: yetkilendirme bayragi, 404 ve dosya adi
  sanitizasyonu.

Hicbir canli API cagrisi yapilmaz.
"""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag_tr.api import routes
from rag_tr.api.main import create_app
from rag_tr.config import Settings
from rag_tr.contracts import ErrorCode
from rag_tr.retrieval.keyword_search import BM25Index
from rag_tr.retrieval.vector_store import make_chunk_id
from rag_tr.service import RAGService


# --- gercek servis, sahte kenarlar -------------------------------------------


class _FakeEmbeddingModel:
    """Deterministik, API'siz embedding: metin uzunlugundan uretilir."""

    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        seed = sum(ord(ch) for ch in text) % 97 + 1
        return [((seed * (i + 1)) % 13) / 13 for i in range(self.dimensions)]

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vector(text) for text in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.array(self._vector(text), dtype=np.float32)


class _MemoryVectorStore:
    """Chroma'nin bu testlerde kullanilan yuzeyi, bellekte.

    Kosinus benzerligi gercekten hesaplanir: silinen bir dokumanin retrieval'dan
    da dustugunu kanitlayan test, calisan bir sorgu yolu gerektiriyor."""

    def __init__(self) -> None:
        self._chunks: list = []
        self._vectors: list = []

    def add(self, chunks, embeddings) -> None:
        for chunk, vector in zip(chunks, np.asarray(embeddings)):
            self._chunks.append(chunk)
            self._vectors.append(np.asarray(vector, dtype=np.float32))

    def delete_by_source(self, source_file: str) -> None:
        kept = [
            (chunk, vector)
            for chunk, vector in zip(self._chunks, self._vectors)
            if chunk.source_file != source_file
        ]
        self._chunks = [chunk for chunk, _ in kept]
        self._vectors = [vector for _, vector in kept]

    def count(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list:
        return list(self._chunks)

    def query(self, query_embedding, top_k: int, source_file: str | None = None):
        query_vector = np.asarray(query_embedding, dtype=np.float32)
        scored = []
        for chunk, vector in zip(self._chunks, self._vectors):
            if source_file is not None and chunk.source_file != source_file:
                continue
            denominator = (np.linalg.norm(query_vector) * np.linalg.norm(vector)) or 1.0
            score = float(np.dot(query_vector, vector) / denominator)
            scored.append((make_chunk_id(chunk.source_file, chunk.chunk_index), score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def get_chunks(self, chunk_ids: list[str]) -> list:
        wanted = set(chunk_ids)
        return [
            chunk
            for chunk in self._chunks
            if make_chunk_id(chunk.source_file, chunk.chunk_index) in wanted
        ]


@pytest.fixture
def service(tmp_path) -> RAGService:
    settings = Settings(_env_file=None, chroma_persist_dir=str(tmp_path / "chroma"))
    return RAGService(
        settings,
        embedding_model=_FakeEmbeddingModel(),
        vector_store=_MemoryVectorStore(),
        bm25_index=BM25Index(),
    )


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- servis: listeleme --------------------------------------------------------


def test_an_empty_corpus_lists_no_documents(service):
    assert service.list_documents() == []


def test_documents_are_listed_with_their_chunk_counts(service, tmp_path):
    service.ingest_files(
        [
            _write(tmp_path, "tarih.md", "Osmanlı Devleti 1299 yılında kuruldu."),
            _write(tmp_path, "cografya.md", "Türkiye'nin başkenti Ankara'dır."),
        ]
    )

    documents = service.list_documents()

    assert [d.source_file for d in documents] == ["cografya.md", "tarih.md"], "ada göre sıralı"
    assert all(d.chunk_count >= 1 for d in documents)
    assert sum(d.chunk_count for d in documents) == service.vector_store.count()


# --- servis: silme ------------------------------------------------------------


def test_removing_a_document_clears_it_from_both_indexes(service, tmp_path):
    """Asil risk burada: yalnizca vektor deposundan silmek, anahtar kelime
    aramasinin kaldirilmis bir dokumani dondurmeye devam etmesi demekti."""
    service.ingest_files(
        [
            _write(tmp_path, "tarih.md", "Osmanlı Devleti 1299 yılında kuruldu."),
            _write(tmp_path, "cografya.md", "Türkiye'nin başkenti Ankara'dır."),
        ]
    )
    before = service.bm25_index.size()

    removed = service.remove_document("tarih.md")

    assert removed >= 1
    assert [d.source_file for d in service.list_documents()] == ["cografya.md"]
    assert service.bm25_index.size() == before - removed, "BM25 de küçülmeli"
    assert all(c.source_file != "tarih.md" for c in service.vector_store.all_chunks())


def test_removing_an_unknown_document_changes_nothing(service, tmp_path):
    service.ingest_files([_write(tmp_path, "tarih.md", "Osmanlı Devleti 1299.")])
    before = service.vector_store.count()

    assert service.remove_document("yok.md") == 0
    assert service.vector_store.count() == before


def test_resetting_empties_both_indexes(service, tmp_path):
    service.ingest_files(
        [
            _write(tmp_path, "tarih.md", "Osmanlı Devleti 1299 yılında kuruldu."),
            _write(tmp_path, "cografya.md", "Türkiye'nin başkenti Ankara'dır."),
        ]
    )

    before = service.vector_store.count()

    removed = service.reset_corpus()

    assert removed == before, "silinen chunk sayısı korpusun tamamı olmalı"
    assert service.list_documents() == []
    assert service.vector_store.count() == 0
    assert service.bm25_index.size() == 0


def test_a_document_removed_from_the_corpus_is_no_longer_retrievable(service, tmp_path):
    service.ingest_files([_write(tmp_path, "tarih.md", "Osmanlı Devleti 1299 yılında kuruldu.")])
    assert service.retrieve("Osmanlı Devleti ne zaman kuruldu", top_k=3).passages

    service.remove_document("tarih.md")

    assert service.retrieve("Osmanlı Devleti ne zaman kuruldu", top_k=3).passages == []


# --- HTTP katmani -------------------------------------------------------------


WRITE_TOKEN = "test-write-token"
WRITE_HEADERS = {"Authorization": f"Bearer {WRITE_TOKEN}"}


class _StubSettings:
    embedding_model_name = "fake-embed-model"
    top_k_final = 5
    allowed_origins = ""
    upload_enabled = True

    def __init__(self, token: str | None) -> None:
        self.ingest_api_token = token
        self.upload_max_bytes = 1024


class _StubVectorStore:
    def __init__(self, count: int) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class _StubBM25:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _StubService:
    def __init__(self, *, token: str | None = WRITE_TOKEN, documents=None) -> None:
        from rag_tr.service import DocumentSummary

        self.settings = _StubSettings(token)
        self._documents = documents if documents is not None else [
            DocumentSummary(source_file="tarih.md", chunk_count=3),
            DocumentSummary(source_file="cografya.md", chunk_count=2),
        ]
        self.vector_store = _StubVectorStore(sum(d.chunk_count for d in self._documents))
        self.bm25_index = _StubBM25(sum(d.chunk_count for d in self._documents))
        self.removed: list[str] = []
        self.reset_called = False

    def list_documents(self):
        return list(self._documents)

    def remove_document(self, source_file: str) -> int:
        self.removed.append(source_file)
        for document in self._documents:
            if document.source_file == source_file:
                self._documents = [d for d in self._documents if d is not document]
                return document.chunk_count
        return 0

    def reset_corpus(self) -> int:
        self.reset_called = True
        removed = sum(d.chunk_count for d in self._documents)
        self._documents = []
        return removed


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Yukleme dizini testte gecici bir yere bakar; gercek data/uploads'a
    # dokunulmaz. Modul uzerinden yamalaniyor cunku route'lar onu oyle okuyor.
    monkeypatch.setattr(routes, "UPLOAD_DIR", tmp_path / "uploads")
    (tmp_path / "uploads").mkdir()

    def build(**kwargs):
        service = _StubService(**kwargs)
        # Yazma uclari token ister; okuma testleri de ayni istemciyi
        # kullaniyor cunku listeleme basligi yok sayar.
        client = TestClient(create_app(service=service), headers=WRITE_HEADERS)
        return client, service

    return build


def test_listing_is_available_without_any_token(client):
    service = _StubService()
    local = TestClient(create_app(service=service))

    response = local.get("/documents")

    assert response.status_code == 200
    body = response.json()
    assert [d["source_file"] for d in body["documents"]] == ["tarih.md", "cografya.md"]
    assert body["total_chunks"] == 5
    assert body["keyword_index_size"] == 5


def test_listing_reports_a_read_only_corpus_when_no_token_is_configured(client):
    local, _ = client(token=None)

    body = local.get("/documents").json()

    assert body["can_modify"] is False, "arayüz silme düğmesi göstermemeli"


# --- can_modify: ISTEGE gore hesaplanir, sunucu yapilandirmasina gore degil ---


def test_can_modify_is_false_for_a_caller_without_a_token(client):
    """Sunucuda token TANIMLI ama cagiran onu sunmuyor.

    Bu alan sunucunun token'i olup olmadigini degil, cagiranin gercekten
    silebilecegini bildirmeli. Aksi halde dagitimda anonim bir tarayiciya
    silme dugmesi gosterilir ve dugme 401 ile donerdi."""
    service = _StubService()  # token tanimli
    local = TestClient(create_app(service=service))

    body = local.get("/documents").json()

    assert body["can_modify"] is False


def test_can_modify_is_true_for_a_caller_with_a_valid_token(client):
    local, _ = client()  # istemci WRITE_HEADERS tasiyor

    body = local.get("/documents").json()

    assert body["can_modify"] is True


def test_can_modify_is_false_for_a_caller_with_a_wrong_token(client):
    service = _StubService()
    local = TestClient(
        create_app(service=service), headers={"Authorization": "Bearer yanlis"}
    )

    body = local.get("/documents").json()

    assert body["can_modify"] is False


def test_can_modify_agrees_with_what_deletion_actually_does(client):
    """Alan ile davranis birbirini tutmali: iki istemci, iki sonuc."""
    service = _StubService()
    anonymous = TestClient(create_app(service=service))
    authorised = TestClient(create_app(service=service), headers=WRITE_HEADERS)

    assert anonymous.get("/documents").json()["can_modify"] is False
    assert anonymous.delete("/documents/tarih.md").status_code == 401

    assert authorised.get("/documents").json()["can_modify"] is True
    assert authorised.delete("/documents/tarih.md").status_code == 200


def test_a_document_can_be_removed_with_a_valid_token(client):
    local, service = client()

    response = local.delete("/documents/tarih.md")

    assert response.status_code == 200
    assert response.json()["chunks_removed"] == 3
    assert service.removed == ["tarih.md"]


def test_removing_without_a_token_is_refused(client):
    service = _StubService()
    local = TestClient(create_app(service=service))

    response = local.delete("/documents/tarih.md")

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == ErrorCode.UNAUTHORIZED.value
    assert service.removed == [], "yetki yokken servise hiç dokunulmamalı"


def test_removing_is_closed_when_no_token_is_configured(client):
    local, service = client(token=None)

    response = local.delete("/documents/tarih.md")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == ErrorCode.INGEST_DISABLED.value
    assert service.removed == []


def test_removing_an_unknown_document_is_a_404_with_its_own_code(client):
    local, _ = client()

    response = local.delete("/documents/olmayan.md")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == ErrorCode.DOCUMENT_NOT_FOUND.value


def test_a_path_traversing_name_is_sanitised_before_deletion(client):
    """`../` tasiyan bir ad yukleme yolundaki ile ayni fonksiyondan gecer."""
    local, service = client()

    local.delete("/documents/..%2F..%2Fetc%2Fpasswd")

    assert all(".." not in name and "/" not in name for name in service.removed)


def test_resetting_empties_the_corpus(client):
    local, service = client()

    response = local.post("/documents/reset")

    assert response.status_code == 200
    assert response.json()["chunks_removed"] == 5
    assert service.reset_called is True


def test_resetting_without_a_token_is_refused(client):
    service = _StubService()
    local = TestClient(create_app(service=service))

    response = local.post("/documents/reset")

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == ErrorCode.UNAUTHORIZED.value
    assert service.reset_called is False


def test_removing_also_deletes_the_uploaded_copy(client, tmp_path):
    local, _ = client()
    copy = tmp_path / "uploads" / "tarih.md"
    copy.write_text("içerik", encoding="utf-8")

    local.delete("/documents/tarih.md")

    assert not copy.exists(), "diskteki kopya da kaldırılmalı"


def test_reset_clears_the_upload_directory(client, tmp_path):
    local, _ = client()
    for name in ("tarih.md", "cografya.md"):
        (tmp_path / "uploads" / name).write_text("içerik", encoding="utf-8")

    local.post("/documents/reset")

    assert list((tmp_path / "uploads").glob("*")) == []


def test_the_token_value_is_never_echoed_back(client):
    """Yanitlar token'i ya da adini tasimamali."""
    local, _ = client()

    listing = local.get("/documents")
    removal = local.delete("/documents/tarih.md")

    for response in (listing, removal):
        assert WRITE_TOKEN not in response.text
        assert "INGEST_API_TOKEN" not in response.text
