"""OpenAPI belgesindeki dosya alanlari.

Gercek bir hatadan geliyor: Swagger UI'da `/upload` dosya secme alani yerine
duz bir metin kutusu gosteriyordu (`files: array<string>`) ve arayuzden PDF
yuklenemiyordu. Sebep calisma zamani degil dokumantasyondu -- FastAPI
(Pydantic v2) ikili alani `contentMediaType` ile yaziyor, Swagger UI ise
dosya secicisini `format: "binary"` gorunce ciziyor.

Bu testler iki seyi birden tutuyor: semanin Swagger'in tanidigi bicimde
kalmasi ve uc noktalarin calisma zamani davranisinin degismemis olmasi.
"""

import pytest
from fastapi.testclient import TestClient

from rag_tr.api.main import create_app
from rag_tr.config import Settings


class _StubVectorStore:
    def count(self) -> int:
        return 0


class _StubBM25:
    def size(self) -> int:
        return 0


class _StubService:
    """Sema uretimi icin yeterli en kucuk yuzey; hicbir sey cagrilmaz."""

    def __init__(self) -> None:
        self.settings = Settings(_env_file=None)
        self.vector_store = _StubVectorStore()
        self.bm25_index = _StubBM25()


@pytest.fixture(scope="module")
def spec() -> dict:
    return create_app(service=_StubService()).openapi()


def _request_body_schema(spec: dict, path: str) -> dict:
    content = spec["paths"][path]["post"]["requestBody"]["content"]
    assert "multipart/form-data" in content, f"{path}: multipart gövde bekleniyor"
    reference = content["multipart/form-data"]["schema"]["$ref"]
    return spec["components"]["schemas"][reference.split("/")[-1]]


UPLOAD_PATHS = ("/upload", "/ingest")


@pytest.mark.parametrize("path", UPLOAD_PATHS)
def test_file_field_is_declared_as_binary(spec, path):
    """Swagger UI dosya secicisini yalnizca bu bicimde ciziyor."""
    files = _request_body_schema(spec, path)["properties"]["files"]

    assert files["type"] == "array", "birden fazla dosya kabul edilmeli"
    assert files["items"] == {"type": "string", "format": "binary"}


@pytest.mark.parametrize("path", UPLOAD_PATHS)
def test_file_field_is_required(spec, path):
    assert "files" in _request_body_schema(spec, path).get("required", [])


def test_no_binary_field_is_left_in_the_2020_12_form(spec):
    """Sema geneli taranir: baska bir dosya alani eklenirse de yakalanir."""
    found: list[str] = []

    def walk(node, path="$"):
        if isinstance(node, dict):
            if node.get("contentMediaType") == "application/octet-stream":
                found.append(path)
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")

    walk(spec)

    assert found == [], f"Swagger'ın tanımadığı ikili alan kaldı: {found}"


def test_every_endpoint_is_still_documented(spec):
    """Sema yeniden yazilirken hicbir uc nokta dusmemeli."""
    assert set(spec["paths"]) == {
        "/health",
        "/query",
        "/ingest",
        "/upload",
        "/agent/ask",
        "/documents",
        "/documents/reset",
        "/documents/{source_file}",
    }


def test_json_endpoints_are_untouched(spec):
    """Duzeltme yalnizca ikili alanlara dokunmali."""
    ask = spec["paths"]["/agent/ask"]["post"]["requestBody"]["content"]

    assert "application/json" in ask
    reference = ask["application/json"]["schema"]["$ref"]
    body = spec["components"]["schemas"][reference.split("/")[-1]]
    assert "question" in body["properties"]
    assert "language" in body["properties"]


def test_the_schema_is_built_once_and_cached():
    """`openapi()` her istekte yeniden uretilmemeli."""
    app = create_app(service=_StubService())

    first = app.openapi()
    second = app.openapi()

    assert first is second


# --- calisma zamani davranisi degismedi ---------------------------------------


def test_upload_still_rejects_an_empty_file_list(tmp_path, monkeypatch):
    """Dokumantasyon duzeltmesi uc noktanin davranisini degistirmemeli."""
    from rag_tr.api import routes

    monkeypatch.setattr(routes, "UPLOAD_DIR", tmp_path / "uploads")

    settings = Settings(_env_file=None, upload_enabled=True)

    class _Service(_StubService):
        def __init__(self) -> None:
            super().__init__()
            self.settings = settings

    client = TestClient(create_app(service=_Service()))

    # Dosya alani zorunlu: hic dosya gonderilmeyen istek dogrulamaya takilir.
    response = client.post("/upload")

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"
