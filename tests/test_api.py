"""FastAPI katmani testleri.

`rag_tr.api.main` import etmek artik uygulamayi kurmuyor; create_app() enjekte
edilmis bir servis aliyor, bu yuzden bu testler ne embedding modeli indiriyor ne
de Anthropic API key'i istiyor.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_tr.api import main as main_module
from rag_tr.api import routes
from rag_tr.api.main import create_app
from rag_tr.service import IngestResult
from rag_tr.generation.answerer import AnswerResult


class _StubVectorStore:
    def __init__(self, count: int = 7) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class _StubBM25:
    def __init__(self, size: int = 5) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _StubSettings:
    embedding_model_name = "fake-embed-model"
    top_k_final = 5


class StubService:
    """Routes'un dokundugu yuzeyin tamami: settings, vector_store, bm25_index,
    ingest_files, query."""

    def __init__(self, query_error: Exception | None = None) -> None:
        self.settings = _StubSettings()
        self.vector_store = _StubVectorStore()
        self.bm25_index = _StubBM25()
        self.query_error = query_error
        self.ingested_paths: list[list[Path]] = []
        self.queries: list[tuple[str, int]] = []

    def ingest_files(self, saved_paths):
        self.ingested_paths.append(list(saved_paths))
        return IngestResult(
            ingested_files=[p.name for p in saved_paths],
            failed_files=[],
            chunk_count=3 * len(saved_paths),
        )

    def query(self, question: str, top_k: int) -> AnswerResult:
        self.queries.append((question, top_k))
        if self.query_error is not None:
            raise self.query_error
        return AnswerResult(
            answer="Ankara'dır [1].",
            sources=[
                {"index": 1, "source_file": "cografya.md", "page_number": None, "text": "Ankara"}
            ],
            used_chunk_ids=["cografya.md::0"],
        )


@pytest.fixture
def service() -> StubService:
    return StubService()


@pytest.fixture
def client(service, tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setattr(routes, "UPLOAD_DIR", tmp_path / "uploads")
    return TestClient(create_app(service=service))


def test_importing_main_does_not_build_the_app():
    """Modulu import etmek Settings()/EmbeddingModel()/Anthropic() kurmamali."""
    assert "app" not in vars(main_module)
    assert callable(main_module.create_app)


def test_health_reports_index_sizes(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "embedding_model": "fake-embed-model",
        "chunk_count": 7,
        "keyword_index_size": 5,
    }


def test_ingest_saves_file_and_returns_counts(client, service, tmp_path):
    response = client.post("/ingest", files={"files": ("notlar.txt", b"merhaba dunya", "text/plain")})

    assert response.status_code == 200
    body = response.json()
    assert body["ingested_files"] == ["notlar.txt"]
    assert body["failed_files"] == []
    assert body["chunk_count"] == 3
    saved = service.ingested_paths[0][0]
    assert saved.name == "notlar.txt"
    assert saved.read_bytes() == b"merhaba dunya"


@pytest.mark.parametrize(
    "filename",
    [
        "../../../../etc/parola.txt",
        "..\\..\\..\\windows\\sistem.txt",
        "/tmp/mutlak.txt",
        "C:\\Windows\\yol.txt",
    ],
)
def test_ingest_strips_path_components_from_filename(client, service, tmp_path, filename):
    response = client.post("/ingest", files={"files": (filename, b"veri", "text/plain")})

    assert response.status_code == 200
    saved = service.ingested_paths[0][0]
    assert saved.parent == (tmp_path / "uploads").resolve()
    assert ".." not in saved.parts


def test_ingest_rejects_unsupported_extension(client, service):
    response = client.post("/ingest", files={"files": ("kotu.exe", b"MZ", "application/octet-stream")})

    assert response.status_code == 400
    assert service.ingested_paths == []


def test_ingest_rejects_empty_or_dotted_filename(client, service):
    response = client.post("/ingest", files={"files": ("..", b"veri", "text/plain")})

    assert response.status_code == 400
    assert service.ingested_paths == []


def test_query_returns_answer_sources_and_chunk_ids(client, service):
    response = client.post("/query", json={"question": "Başkent neresi?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Ankara'dır [1]."
    assert body["used_chunk_ids"] == ["cografya.md::0"]
    assert body["sources"][0]["source_file"] == "cografya.md"
    assert service.queries == [("Başkent neresi?", 5)], "top_k verilmezse varsayilan kullanilmali"


def test_query_honours_explicit_top_k(client, service):
    client.post("/query", json={"question": "Başkent neresi?", "top_k": 2})

    assert service.queries == [("Başkent neresi?", 2)]


def test_module_level_app_attribute_still_works_for_uvicorn(monkeypatch, service):
    """`uvicorn rag_tr.api.main:app` komutu calismaya devam etmeli, ama app
    yalnizca bu attribute'a erisildiginde kurulmali."""
    monkeypatch.setattr(main_module, "build_service", lambda: service)
    try:
        app = main_module.app
        assert app.state.service is service
        assert TestClient(app).get("/health").status_code == 200
    finally:
        vars(main_module).pop("app", None)


def test_unknown_module_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        main_module.bulunmayan_attribute
