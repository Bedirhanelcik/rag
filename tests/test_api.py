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
from rag_tr.contracts import (
    NO_CONTEXT_MESSAGE,
    ErrorCode,
    GenerationError,
    RetrievalError,
)
from rag_tr.generation.answerer import AnswerResult, no_context_result
from rag_tr.service import IngestResult


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


INGEST_TOKEN = "test-ingest-token"
INGEST_HEADERS = {"Authorization": f"Bearer {INGEST_TOKEN}"}


class _StubSettings:
    embedding_model_name = "fake-embed-model"
    top_k_final = 5
    ingest_api_token = INGEST_TOKEN


class StubService:
    """Routes'un dokundugu yuzeyin tamami: settings, vector_store, bm25_index,
    ingest_files, query."""

    def __init__(self, query_error: Exception | None = None) -> None:
        self.settings = _StubSettings()
        self.vector_store = _StubVectorStore()
        self.bm25_index = _StubBM25()
        self.query_error = query_error
        self.query_result: AnswerResult | None = None
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
        if self.query_result is not None:
            return self.query_result
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
    response = client.post("/ingest", files={"files": ("notlar.txt", b"merhaba dunya", "text/plain")}, headers=INGEST_HEADERS)

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
    response = client.post("/ingest", files={"files": (filename, b"veri", "text/plain")}, headers=INGEST_HEADERS)

    assert response.status_code == 200
    saved = service.ingested_paths[0][0]
    assert saved.parent == (tmp_path / "uploads").resolve()
    assert ".." not in saved.parts


def test_ingest_rejects_unsupported_extension(client, service):
    response = client.post("/ingest", files={"files": ("kotu.exe", b"MZ", "application/octet-stream")}, headers=INGEST_HEADERS)

    assert response.status_code == 400
    assert service.ingested_paths == []


def test_ingest_rejects_empty_or_dotted_filename(client, service):
    response = client.post("/ingest", files={"files": ("..", b"veri", "text/plain")}, headers=INGEST_HEADERS)

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


def test_query_response_carries_machine_readable_status(client):
    body = client.post("/query", json={"question": "Başkent neresi?"}).json()

    assert body["status"] == "answered"


def test_query_reports_no_relevant_context_status(tmp_path, monkeypatch):
    service = StubService()
    service.query_result = no_context_result()
    monkeypatch.setattr(routes, "UPLOAD_DIR", tmp_path / "uploads")
    local_client = TestClient(create_app(service=service))

    response = local_client.post("/query", json={"question": "Bilinmeyen bir sey?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_relevant_context"
    assert body["answer"] == NO_CONTEXT_MESSAGE
    assert body["sources"] == []
    assert body["used_chunk_ids"] == []


def test_generation_error_maps_to_502_with_code(tmp_path, monkeypatch):
    service = StubService(query_error=GenerationError("Claude down"))
    monkeypatch.setattr(routes, "UPLOAD_DIR", tmp_path / "uploads")
    local_client = TestClient(create_app(service=service))

    response = local_client.post("/query", json={"question": "Soru?"})

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == ErrorCode.GENERATION_ERROR.value


def test_retrieval_error_maps_to_500_with_code(tmp_path, monkeypatch):
    service = StubService(query_error=RetrievalError("chroma down"))
    monkeypatch.setattr(routes, "UPLOAD_DIR", tmp_path / "uploads")
    local_client = TestClient(create_app(service=service))

    response = local_client.post("/query", json={"question": "Soru?"})

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == ErrorCode.RETRIEVAL_ERROR.value


def test_unsupported_extension_reports_its_error_code(client):
    response = client.post("/ingest", files={"files": ("kotu.exe", b"MZ", "application/octet-stream")}, headers=INGEST_HEADERS)

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == ErrorCode.UNSUPPORTED_FILE_TYPE.value


def test_invalid_filename_reports_its_error_code(client):
    response = client.post("/ingest", files={"files": ("..", b"veri", "text/plain")}, headers=INGEST_HEADERS)

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == ErrorCode.INVALID_FILENAME.value


def test_top_k_zero_is_rejected_instead_of_silently_defaulting(client, service):
    response = client.post("/query", json={"question": "Soru?", "top_k": 0})

    assert response.status_code == 422
    assert service.queries == []


def test_empty_question_is_rejected(client, service):
    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422
    assert service.queries == []


# --- /ingest kimlik dogrulamasi ---


def test_ingest_without_a_token_is_rejected(client, service):
    response = client.post("/ingest", files={"files": ("a.txt", b"veri")})

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == ErrorCode.UNAUTHORIZED.value
    assert service.ingested_paths == [], "yetkisiz istek ingestion'a ulasmamali"


@pytest.mark.parametrize(
    "header",
    [
        {"Authorization": "Bearer yanlis-token"},
        {"Authorization": f"Basic {INGEST_TOKEN}"},
        {"Authorization": INGEST_TOKEN},
        {"Authorization": "Bearer "},
        {"X-Ingest-Token": INGEST_TOKEN},
    ],
)
def test_ingest_rejects_a_wrong_or_malformed_token(client, service, header):
    response = client.post("/ingest", files={"files": ("a.txt", b"veri")}, headers=header)

    assert response.status_code == 401
    assert service.ingested_paths == []


def test_ingest_accepts_the_configured_token(client, service):
    response = client.post(
        "/ingest", files={"files": ("a.txt", b"veri")}, headers=INGEST_HEADERS
    )

    assert response.status_code == 200
    assert [path.name for path in service.ingested_paths[0]] == ["a.txt"]


def test_ingest_is_closed_when_no_token_is_configured(tmp_path, monkeypatch):
    """Guvenli varsayilan: token tanimlanmamissa uc nokta hic hizmet vermez,
    yapilandirilmamis bir dagitimda korpusa herkes yazamaz."""

    class _NoTokenSettings(_StubSettings):
        ingest_api_token = None

    service = StubService()
    service.settings = _NoTokenSettings()
    monkeypatch.setattr(routes, "UPLOAD_DIR", tmp_path / "uploads")
    local = TestClient(create_app(service=service))

    response = local.post("/ingest", files={"files": ("a.txt", b"veri")})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == ErrorCode.INGEST_DISABLED.value
    assert service.ingested_paths == []


def test_ingest_token_defaults_to_unset_in_settings():
    from rag_tr.config import Settings

    assert Settings(_env_file=None).ingest_api_token is None
