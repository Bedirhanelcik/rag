"""POST /upload testleri.

Gercek embedding modeli, Chroma, Gemini veya Anthropic kullanilmaz: servis
fake'lenir, yalnizca HTTP katmani ve bayrak/sinir davranisi sinanir.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_tr.api import routes
from rag_tr.api.main import create_app
from rag_tr.contracts import ErrorCode
from rag_tr.service import IngestResult


INGEST_TOKEN = "test-ingest-token"
INGEST_HEADERS = {"Authorization": f"Bearer {INGEST_TOKEN}"}


class _StubSettings:
    embedding_model_name = "fake-embed-model"
    top_k_final = 5
    ingest_api_token = INGEST_TOKEN

    def __init__(self, *, upload_enabled: bool = True, upload_max_bytes: int = 1024) -> None:
        self.upload_enabled = upload_enabled
        self.upload_max_bytes = upload_max_bytes


class _StubVectorStore:
    def __init__(self, count: int = 0) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class StubService:
    """ingest_files'in gordugu yuzey. Gercek pipeline cagrilmaz."""

    def __init__(
        self,
        *,
        upload_enabled: bool = True,
        upload_max_bytes: int = 1024,
        chunks_per_file: int = 3,
        fail: bool = False,
        error: Exception | None = None,
    ) -> None:
        self.settings = _StubSettings(
            upload_enabled=upload_enabled, upload_max_bytes=upload_max_bytes
        )
        self.vector_store = _StubVectorStore()
        self.calls: list[list[Path]] = []
        self._chunks = chunks_per_file
        self._fail = fail
        self._error = error

    def ingest_files(self, saved_paths: list[Path]) -> IngestResult:
        self.calls.append(list(saved_paths))
        if self._error is not None:
            raise self._error
        names = [path.name for path in saved_paths]
        if self._fail:
            return IngestResult(ingested_files=[], failed_files=names, chunk_count=0)
        self.vector_store._count += self._chunks * len(names)
        return IngestResult(
            ingested_files=names, failed_files=[], chunk_count=self._chunks * len(names)
        )


@pytest.fixture
def upload_dir(tmp_path, monkeypatch) -> Path:
    target = tmp_path / "uploads"
    monkeypatch.setattr(routes, "UPLOAD_DIR", target)
    return target


def _client(service: StubService) -> TestClient:
    """Yetkili istemci.

    Token basligi varsayilan olarak ekleniyor: bu dosyadaki testlerin konusu
    yetkilendirme degil, yukleme davranisi (tur, boyut, ad sanitizasyonu).
    Yetki kontrolu kendi testlerinde ayrica sinanir."""
    return TestClient(
        create_app(service=service),
        raise_server_exceptions=False,
        headers=INGEST_HEADERS,
    )


def _anonymous_client(service: StubService) -> TestClient:
    return TestClient(create_app(service=service), raise_server_exceptions=False)


# --- kabul edilen turler ---


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("notlar.txt", b"merhaba dunya"),
        ("belge.md", b"# baslik"),
        ("rapor.pdf", b"%PDF-1.4 sahte"),
    ],
)
def test_supported_types_are_accepted(upload_dir, filename, payload):
    service = StubService()

    response = _client(service).post("/upload", files={"files": (filename, payload)})

    assert response.status_code == 200
    body = response.json()
    assert body["files"] == [{"filename": filename, "chunks_created": 3, "status": "ingested"}]
    assert body["total_chunks"] == 3
    assert (upload_dir / filename).read_bytes() == payload


def test_multiple_files_report_per_file_results(upload_dir):
    service = StubService()

    response = _client(service).post(
        "/upload",
        files=[
            ("files", ("bir.txt", b"a")),
            ("files", ("iki.md", b"b")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert [entry["filename"] for entry in body["files"]] == ["bir.txt", "iki.md"]
    assert all(entry["chunks_created"] == 3 for entry in body["files"])
    assert body["total_chunks"] == 6
    # Dosya basina sonuc icin pipeline dosya dosya cagrilir.
    assert len(service.calls) == 2


def test_response_reports_corpus_size_so_the_document_is_queryable(upload_dir):
    service = StubService(chunks_per_file=4)

    body = _client(service).post("/upload", files={"files": ("a.txt", b"x")}).json()

    assert body["corpus_chunks"] == 4


# --- reddedilen turler ---


@pytest.mark.parametrize("filename", ["kotu.exe", "resim.png", "arsiv.zip", "script.sh"])
def test_unsupported_extensions_are_rejected(upload_dir, filename):
    service = StubService()

    response = _client(service).post("/upload", files={"files": (filename, b"veri")})

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == ErrorCode.UNSUPPORTED_FILE_TYPE.value
    assert service.calls == []


def test_empty_or_dotted_filename_is_rejected(upload_dir):
    service = StubService()

    response = _client(service).post("/upload", files={"files": ("..", b"veri")})

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == ErrorCode.INVALID_FILENAME.value
    assert service.calls == []


# --- boyut siniri ---


def test_file_over_the_limit_is_rejected(upload_dir):
    service = StubService(upload_max_bytes=16)

    response = _client(service).post("/upload", files={"files": ("buyuk.txt", b"x" * 64)})

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == ErrorCode.FILE_TOO_LARGE.value
    assert service.calls == [], "sinir asildiginda ingestion hic calismamali"


def test_file_at_the_limit_is_accepted(upload_dir):
    service = StubService(upload_max_bytes=16)

    response = _client(service).post("/upload", files={"files": ("tam.txt", b"x" * 16)})

    assert response.status_code == 200


# --- path traversal ---


@pytest.mark.parametrize(
    "filename",
    [
        "../../../../etc/parola.txt",
        "..\\..\\..\\windows\\sistem.txt",
        "/tmp/mutlak.txt",
        "C:\\Windows\\yol.txt",
    ],
)
def test_path_components_are_stripped(upload_dir, filename):
    service = StubService()

    response = _client(service).post("/upload", files={"files": (filename, b"veri")})

    assert response.status_code == 200
    saved = service.calls[0][0]
    assert saved.parent == upload_dir.resolve()
    assert ".." not in saved.parts
    # Yalnizca yukleme dizininde dosya olusmus olmali.
    assert [path.name for path in upload_dir.iterdir()] == [saved.name]


# --- feature flag ---


def test_upload_without_a_token_is_rejected_and_never_ingests(upload_dir):
    """Yetkisiz istek ingestion'a hic ulasmamali."""
    service = StubService()

    response = _anonymous_client(service).post(
        "/upload", files={"files": ("notlar.txt", b"veri")}
    )

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == ErrorCode.UNAUTHORIZED.value
    assert service.calls == [], "yetki yokken ingestion çağrılmamalı"
    assert list(upload_dir.glob("*")) == [] if upload_dir.exists() else True


def test_upload_with_a_wrong_token_is_rejected(upload_dir):
    service = StubService()
    client = TestClient(
        create_app(service=service),
        raise_server_exceptions=False,
        headers={"Authorization": "Bearer yanlis-token"},
    )

    response = client.post("/upload", files={"files": ("notlar.txt", b"veri")})

    assert response.status_code == 401
    assert service.calls == []


def test_upload_is_closed_when_no_token_is_configured(upload_dir):
    """Token hic tanimli degilse uc nokta hizmet vermez."""
    service = StubService()
    service.settings.ingest_api_token = None

    response = _client(service).post("/upload", files={"files": ("notlar.txt", b"veri")})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == ErrorCode.INGEST_DISABLED.value
    assert service.calls == []


def test_upload_disabled_returns_403_even_with_a_valid_token(upload_dir):
    """`UPLOAD_ENABLED=false` yetkiden BAGIMSIZ bir ortam anahtaridir.

    Gecerli token'la bile yukleme yapilamaz; boylece paylasimli bir dagitim
    token iptal etmeye gerek kalmadan salt okunur yapilabilir."""
    service = StubService(upload_enabled=False)

    response = _client(service).post("/upload", files={"files": ("notlar.txt", b"veri")})

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == ErrorCode.UPLOAD_DISABLED.value
    assert service.calls == [], "bayrak kapaliyken ingestion çağrılmamalı"
    assert not upload_dir.exists() or list(upload_dir.iterdir()) == []


def test_authorization_is_checked_before_the_upload_flag(upload_dir):
    """Yetkisiz cagiran, dagitimin bayrak durumunu ogrenmemeli: her hâlde 401.

    Bayrak kontrolu once yapilsaydi anonim bir istek 403 alir ve bu, token'i
    olmayan birine ortam yapilandirmasi hakkinda bilgi verirdi."""
    service = StubService(upload_enabled=False)

    response = _anonymous_client(service).post(
        "/upload", files={"files": ("notlar.txt", b"veri")}
    )

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == ErrorCode.UNAUTHORIZED.value
    assert service.calls == []


def test_upload_is_disabled_by_default_in_settings():
    from rag_tr.config import Settings

    settings = Settings(anthropic_api_key="")

    assert settings.upload_enabled is False, "dagitimda kapali olmali"
    assert settings.upload_max_bytes > 0


# --- ingestion sonucu ---


def test_failed_ingestion_is_reported_per_file(upload_dir):
    service = StubService(fail=True)

    response = _client(service).post("/upload", files={"files": ("bozuk.pdf", b"veri")})

    assert response.status_code == 200
    body = response.json()
    assert body["files"] == [
        {"filename": "bozuk.pdf", "chunks_created": 0, "status": "failed"}
    ]
    assert body["total_chunks"] == 0


def test_ingestion_crash_does_not_leak_as_a_success(upload_dir):
    service = StubService(error=RuntimeError("chroma down"))

    response = _client(service).post("/upload", files={"files": ("a.txt", b"veri")})

    assert response.status_code == 500
    assert "chunks_created" not in response.text


# --- mevcut /ingest bozulmadi ---


def test_existing_ingest_endpoint_still_works_and_ignores_the_flag(upload_dir):
    """/ingest davranisi degismedi: upload bayragi onu etkilemez."""
    service = StubService(upload_enabled=False)

    response = _client(service).post(
        "/ingest", files={"files": ("a.txt", b"veri")}, headers=INGEST_HEADERS
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ingested_files"] == ["a.txt"]
    assert body["chunk_count"] == 3
