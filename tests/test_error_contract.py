"""Tum uc noktalarin ayni hata govdesini konustugunu dogrular.

Istemci tek bir sekle bakabilmeli: {"detail": {"code": ..., "message": ...}}.
Hicbir yanitta yigin izi, ic hata mesaji veya kaynak dosya yolu bulunmamali.
"""

import pytest
from fastapi.testclient import TestClient

from rag_tr.api.main import create_app
from rag_tr.contracts import ErrorCode
from rag_tr.service import IngestResult

TOKEN = "test-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


class _Settings:
    embedding_model_name = "fake-embed-model"
    top_k_final = 5
    upload_enabled = False
    upload_max_bytes = 1024
    ingest_api_token = TOKEN
    allowed_origin_list: list[str] = []


class _VectorStore:
    def __init__(self, count: int = 0) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class _BM25:
    def size(self) -> int:
        return 0


class StubService:
    def __init__(self, chunks: int = 0, error: Exception | None = None) -> None:
        self.settings = _Settings()
        self.vector_store = _VectorStore(chunks)
        self.bm25_index = _BM25()
        self._error = error

    def ingest_files(self, saved_paths) -> IngestResult:
        if self._error is not None:
            raise self._error
        return IngestResult(
            ingested_files=[p.name for p in saved_paths], failed_files=[], chunk_count=1
        )


def _client(service: StubService | None = None) -> TestClient:
    return TestClient(create_app(service=service or StubService()), raise_server_exceptions=False)


def _detail(response) -> dict:
    body = response.json()
    assert "detail" in body, f"detail yok: {body}"
    detail = body["detail"]
    assert isinstance(detail, dict), f"detail bir nesne olmali, gelen: {type(detail)}"
    assert sorted(detail) == ["code", "message"], f"beklenmeyen alanlar: {sorted(detail)}"
    assert detail["message"], "mesaj bos olmamali"
    return detail


# --- dogrulama hatalari tek tip --- 


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("dosyasiz ingest", lambda c: c.post("/ingest", headers=HEADERS)),
        ("dosyasiz upload", lambda c: c.post("/upload")),
        ("bos govde", lambda c: c.post("/agent/ask", json={})),
        ("bos soru", lambda c: c.post("/agent/ask", json={"question": ""})),
        ("top_k=0", lambda c: c.post("/agent/ask", json={"question": "s", "top_k": 0})),
    ],
)
def test_validation_errors_use_the_shared_shape(name, call):
    response = call(_client())

    assert response.status_code == 422, name
    detail = _detail(response)
    assert detail["code"] == ErrorCode.INVALID_REQUEST.value


def test_malformed_json_is_a_clean_validation_error():
    response = _client().post(
        "/agent/ask", content=b"{bozuk", headers={"content-type": "application/json"}
    )

    assert response.status_code == 422
    assert _detail(response)["code"] == ErrorCode.INVALID_REQUEST.value


def test_validation_message_does_not_echo_the_raw_pydantic_payload():
    """Ham hata listesi girdiyi ve ic alan yollarini yansitiyordu; artik kisa
    bir ozet donuyor."""
    response = _client().post("/agent/ask", json={"question": "s", "top_k": -5})

    text = response.text
    assert "ctx" not in text
    assert "greater_than_equal" not in text, "pydantic hata tipi sizmamali"
    assert len(_detail(response)["message"]) <= 200


# --- yonlendirme hatalari tek tip ---


def test_unknown_route_uses_the_shared_shape():
    response = _client().post("/boyle-bir-uc-nokta-yok")

    assert response.status_code == 404
    assert _detail(response)["code"] == ErrorCode.NOT_FOUND.value


def test_wrong_method_uses_the_shared_shape():
    response = _client().get("/agent/ask")

    assert response.status_code == 405
    assert _detail(response)["code"] == ErrorCode.METHOD_NOT_ALLOWED.value


# --- kendi hatalarimiz degismeden geciyor ---


@pytest.mark.parametrize(
    ("call", "status", "code"),
    [
        (lambda c: c.post("/agent/ask", json={"question": "s"}), 409, ErrorCode.CORPUS_EMPTY),
        (lambda c: c.post("/ingest", files={"files": ("a.txt", b"x")}), 401, ErrorCode.UNAUTHORIZED),
        # Yazma uclarinin hepsi ayni kapidan geciyor: token yoksa 401.
        (
            lambda c: c.post("/upload", files={"files": ("a.txt", b"x")}),
            401,
            ErrorCode.UNAUTHORIZED,
        ),
        (lambda c: c.delete("/documents/a.txt"), 401, ErrorCode.UNAUTHORIZED),
        (lambda c: c.post("/documents/reset"), 401, ErrorCode.UNAUTHORIZED),
        (
            lambda c: c.post(
                "/ingest", files={"files": ("k.exe", b"MZ")}, headers=HEADERS
            ),
            400,
            ErrorCode.UNSUPPORTED_FILE_TYPE,
        ),
    ],
)
def test_domain_errors_keep_their_codes(call, status, code):
    response = call(_client())

    assert response.status_code == status
    assert _detail(response)["code"] == code.value


# --- beklenmeyen istisna ---


def test_unhandled_exception_leaks_nothing():
    service = StubService(error=RuntimeError("chroma down: /secret/key=abc123"))

    response = _client(service).post(
        "/ingest", files={"files": ("a.txt", b"veri")}, headers=HEADERS
    )

    assert response.status_code == 500
    detail = _detail(response)
    assert detail["code"] == ErrorCode.INTERNAL_ERROR.value
    text = response.text
    for leak in ("Traceback", "chroma down", "secret", "abc123", "rag_tr", ".py"):
        assert leak not in text, f"sizinti: {leak}"


# --- sozlesme ile belge arasindaki tutarlilik ---------------------------------


def test_every_error_code_is_documented_in_the_readme():
    """Her `ErrorCode` README'deki sozlesme tablosunda gecmeli.

    Bu kontrol, kod ile belgenin birbirinden ayri dusmesini yakaliyor: yazma
    uclarinin yetkilendirmesi degistiginde README bir sure eski tasarimi
    anlatmaya devam etmisti. Sozlesmeyi makineyle okunabilir tutma hedefi,
    belgenin de dogru kalmasini gerektiriyor.
    """
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    missing = [code.value for code in ErrorCode if f"`{code.value}`" not in readme]

    assert missing == [], f"README'de belgelenmemiş hata kodları: {missing}"
