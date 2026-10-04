"""Hangi uc noktanin acik, hangisinin korumali oldugu.

Bu dosya tek bir seyi sabitliyor: korpusu DEGISTIREN her uc nokta token
ister, okuyan hicbiri istemez. Yeni bir yazma ucu eklenip korumasiz
birakilirsa buradaki liste testi onu yakalar.

Hicbir canli API cagrisi yapilmaz: servis ve agent fake'lenir.
"""

import pytest
from fastapi.testclient import TestClient

from rag_tr.agent.contracts import AgentResult, AgentStatus
from rag_tr.api import routes
from rag_tr.api.main import create_app
from rag_tr.contracts import ErrorCode
from rag_tr.service import DocumentSummary, IngestResult

TOKEN = "test-write-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


class _Settings:
    embedding_model_name = "fake-embed-model"
    top_k_final = 5
    upload_enabled = True
    upload_max_bytes = 4096
    ingest_api_token = TOKEN
    allowed_origin_list: list[str] = []


class _VectorStore:
    def count(self) -> int:
        return 1


class _BM25:
    def size(self) -> int:
        return 1


class _Service:
    def __init__(self) -> None:
        self.settings = _Settings()
        self.vector_store = _VectorStore()
        self.bm25_index = _BM25()
        self.writes: list[str] = []

    # --- yazma yolu: cagrildiysa kaydedilir ---
    def ingest_files(self, saved_paths):
        self.writes.append("ingest")
        names = [path.name for path in saved_paths]
        return IngestResult(ingested_files=names, failed_files=[], chunk_count=len(names))

    def remove_document(self, source_file: str) -> int:
        self.writes.append("remove")
        return 1

    def reset_corpus(self) -> int:
        self.writes.append("reset")
        return 1

    # --- okuma yolu ---
    def list_documents(self):
        return [DocumentSummary(source_file="a.md", chunk_count=1)]


class _Agent:
    def run(self, question: str) -> AgentResult:
        return AgentResult(
            answer="cevap",
            status=AgentStatus.ANSWERED_WITHOUT_RETRIEVAL,
            sources=[],
            steps=[],
            tool_calls=[],
        )


@pytest.fixture
def service(tmp_path, monkeypatch) -> _Service:
    monkeypatch.setattr(routes, "UPLOAD_DIR", tmp_path / "uploads")
    return _Service()


@pytest.fixture
def anonymous(service) -> TestClient:
    return TestClient(create_app(service=service, agent=_Agent()))


@pytest.fixture
def authorised(service) -> TestClient:
    return TestClient(create_app(service=service, agent=_Agent()), headers=HEADERS)


#: Korpusu degistiren her uc nokta. Yeni bir tanesi eklenirse buraya da
#: eklenmeli; aksi halde "korumasiz yazma ucu" testi onu yakalar.
WRITE_CALLS = {
    "POST /upload": lambda c: c.post("/upload", files={"files": ("a.txt", b"veri")}),
    "POST /ingest": lambda c: c.post("/ingest", files={"files": ("a.txt", b"veri")}),
    "DELETE /documents/{f}": lambda c: c.delete("/documents/a.md"),
    "POST /documents/reset": lambda c: c.post("/documents/reset"),
}

READ_CALLS = {
    "GET /health": lambda c: c.get("/health"),
    "GET /documents": lambda c: c.get("/documents"),
    "POST /agent/ask": lambda c: c.post("/agent/ask", json={"question": "merhaba"}),
}


# --- acik uclar ---------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(READ_CALLS))
def test_read_endpoints_need_no_token(name, anonymous):
    response = READ_CALLS[name](anonymous)

    assert response.status_code == 200, f"{name}: token olmadan çalışmalı"


def test_health_reports_the_corpus_without_a_token(anonymous):
    body = anonymous.get("/health").json()

    assert body["status"] == "ok"
    assert "chunk_count" in body


def test_asking_a_question_needs_no_token(anonymous):
    response = anonymous.post("/agent/ask", json={"question": "merhaba"})

    assert response.status_code == 200
    assert response.json()["answer"] == "cevap"


# --- korumali uclar -----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(WRITE_CALLS))
def test_write_endpoints_reject_a_missing_token(name, anonymous, service):
    response = WRITE_CALLS[name](anonymous)

    assert response.status_code == 401, f"{name}: token olmadan reddedilmeli"
    assert response.json()["detail"]["code"] == ErrorCode.UNAUTHORIZED.value
    assert service.writes == [], f"{name}: yetki yokken servise dokunulmamalı"


@pytest.mark.parametrize("name", sorted(WRITE_CALLS))
def test_write_endpoints_reject_a_wrong_token(name, service):
    client = TestClient(
        create_app(service=service, agent=_Agent()),
        headers={"Authorization": "Bearer yanlis"},
    )

    response = WRITE_CALLS[name](client)

    assert response.status_code == 401
    assert service.writes == []


@pytest.mark.parametrize("name", sorted(WRITE_CALLS))
def test_write_endpoints_reject_a_non_bearer_scheme(name, service):
    client = TestClient(
        create_app(service=service, agent=_Agent()),
        headers={"Authorization": f"Basic {TOKEN}"},
    )

    response = WRITE_CALLS[name](client)

    assert response.status_code == 401
    assert service.writes == []


@pytest.mark.parametrize("name", sorted(WRITE_CALLS))
def test_write_endpoints_accept_a_valid_token(name, authorised):
    response = WRITE_CALLS[name](authorised)

    assert response.status_code == 200, f"{name}: geçerli token ile çalışmalı"


@pytest.mark.parametrize("name", sorted(WRITE_CALLS))
def test_write_endpoints_are_closed_when_no_token_is_configured(name, service):
    """Sunucuda token hic tanimli degilse yazma uclari hizmet vermez."""
    service.settings.ingest_api_token = None
    client = TestClient(create_app(service=service, agent=_Agent()), headers=HEADERS)

    response = WRITE_CALLS[name](client)

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == ErrorCode.INGEST_DISABLED.value
    assert service.writes == []


# --- yetki ile ortam anahtari birbirinden ayri ---------------------------------


def test_only_upload_honours_the_upload_flag(service):
    """`UPLOAD_ENABLED` yalnizca `/upload` icin bir anahtar.

    Silme ve sifirlama yetkiye baglidir, bayraga degil: bayrak kapali diye
    operatorun korpusu toplayamamasi icin bir sebep yok."""
    service.settings.upload_enabled = False
    client = TestClient(create_app(service=service, agent=_Agent()), headers=HEADERS)

    assert client.post("/upload", files={"files": ("a.txt", b"veri")}).status_code == 403
    assert client.post("/ingest", files={"files": ("a.txt", b"veri")}).status_code == 200
    assert client.delete("/documents/a.md").status_code == 200
    assert client.post("/documents/reset").status_code == 200


def test_the_upload_flag_never_grants_access_on_its_own(service):
    """Bayrak acik olmasi yetki ANLAMINA GELMEZ.

    Eski tasarimda bayrak hem yetki hem ortam anahtariydi: acik oldugu anda
    `/upload` ve silme uclari herkese aciliyordu. Bu testin kirilmasi o
    karismanin geri geldigini gosterir."""
    service.settings.upload_enabled = True
    anonymous_client = TestClient(create_app(service=service, agent=_Agent()))

    for name, call in sorted(WRITE_CALLS.items()):
        response = call(anonymous_client)

        assert response.status_code == 401, f"{name}: bayrak açık olsa da token şart"

    assert service.writes == []


# --- listenin kendisi -----------------------------------------------------------


def test_every_mutating_route_is_in_the_protected_list(service):
    """Korumasiz birakilmis bir yazma ucu kalmadigini dogrular.

    Kaynak olarak OpenAPI belgesi kullaniliyor: `app.routes` bu FastAPI
    surumunde dahil edilen router'lari sarmaladigi icin duz bir liste vermiyor,
    belge ise API'nin gercekte disa actigi her yolu ve yontemi tasiyor."""
    app = create_app(service=service, agent=_Agent())

    mutating: set[tuple[str, str]] = {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
        if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}
    }

    protected = {
        ("POST", "/upload"),
        ("POST", "/ingest"),
        ("DELETE", "/documents/{source_file}"),
        ("POST", "/documents/reset"),
    }
    # `/agent/ask` ve eski `/query` yalnizca cevap uretir, korpusu degistirmez.
    read_only_posts = {("POST", "/agent/ask"), ("POST", "/query")}

    assert mutating == protected | read_only_posts, (
        "yeni bir yazma ucu eklendiyse korumalı listeye de eklenmeli: "
        f"{sorted(mutating - protected - read_only_posts)}"
    )
