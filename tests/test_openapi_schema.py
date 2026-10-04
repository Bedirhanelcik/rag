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

    settings = Settings(_env_file=None, upload_enabled=True, ingest_api_token="t")

    class _Service(_StubService):
        def __init__(self) -> None:
            super().__init__()
            self.settings = settings

    client = TestClient(
        create_app(service=_Service()), headers={"Authorization": "Bearer t"}
    )

    # Dosya alani zorunlu: hic dosya gonderilmeyen istek dogrulamaya takilir.
    response = client.post("/upload")

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"


# --- yazma uclarinin yetki semasi ---------------------------------------------
#
# Ayri bir gercek hatadan geliyor: yetki kontrolu `Authorization` basligini elle
# okudugu icin FastAPI bunu kesfedemiyor, belgede guvenlik semasi yazmiyor ve
# Swagger UI'da "Authorize" dugmesi cikmiyordu. Sonuc: `/docs` uzerinden token
# gonderilemiyor, yazma uclarinin hepsi 401 doneyordu ve dagitilmis arayuzden
# denenemiyorlardi.


def test_the_bearer_scheme_is_declared(spec):
    """Swagger UI'nin "Authorize" dugmesini cizmesi icin gereken sema."""
    schemes = spec["components"]["securitySchemes"]

    assert "BearerToken" in schemes
    assert schemes["BearerToken"]["type"] == "http"
    assert schemes["BearerToken"]["scheme"] == "bearer"


@pytest.mark.parametrize(
    "method,path",
    sorted(
        [
            ("post", "/ingest"),
            ("post", "/upload"),
            ("delete", "/documents/{source_file}"),
            ("post", "/documents/reset"),
        ]
    ),
)
def test_every_write_endpoint_requires_the_bearer_scheme(spec, method, path):
    operation = spec["paths"][path][method]

    assert operation.get("security") == [{"BearerToken": []}], (
        f"{method.upper()} {path} belgede token istemiyor görünüyor"
    )


@pytest.mark.parametrize(
    "method,path",
    sorted([("get", "/health"), ("get", "/documents"), ("post", "/agent/ask")]),
)
def test_read_endpoints_declare_no_security(spec, method, path):
    """Okuma uclari token istemiyor; Swagger onlara baslik eklememeli."""
    assert "security" not in spec["paths"][path][method]


def test_every_operation_documented_as_protected_is_really_enforced(tmp_path, monkeypatch):
    """Belge ile davranis ayni olmali -- sabit listeyi degil, GERCEGI olcer.

    Belgede token isteyen her islem, token olmadan cagrildiginda gercekten
    reddedilmeli. Aksi halde Swagger kullanicisina korumali gorunen ama
    aslinda korumasiz bir uc nokta gosterilirdi. Istekler semadan uretiliyor:
    listeyi listeyle karsilastirmak dairesel olurdu.
    """
    from rag_tr.api import routes

    monkeypatch.setattr(routes, "UPLOAD_DIR", tmp_path / "uploads")

    class _TokenService(_StubService):
        def __init__(self) -> None:
            super().__init__()
            self.settings = Settings(_env_file=None, ingest_api_token="bir-token")

        def ingest_files(self, saved_paths):  # pragma: no cover - cagrilmamali
            raise AssertionError("yetkisiz istek ingestion'a ulaşmamalı")

        def remove_document(self, name):  # pragma: no cover - cagrilmamali
            raise AssertionError("yetkisiz istek silmeye ulaşmamalı")

        def reset_corpus(self):  # pragma: no cover - cagrilmamali
            raise AssertionError("yetkisiz istek sıfırlamaya ulaşmamalı")

    app = create_app(service=_TokenService())
    client = TestClient(app)
    spec = app.openapi()

    protected = [
        (method.lower(), path)
        for path, operations in spec["paths"].items()
        for method, operation in operations.items()
        if "security" in operation
    ]
    assert protected, "en az bir korumalı işlem belgelenmiş olmalı"

    for method, path in protected:
        # Yol parametresi varsa somut bir degerle doldurulur.
        url = path.replace("{source_file}", "ornek.md")
        kwargs = {}
        if method == "post" and path in {"/upload", "/ingest"}:
            kwargs["files"] = {"files": ("a.txt", b"veri")}

        response = client.request(method.upper(), url, **kwargs)

        assert response.status_code == 401, (
            f"{method.upper()} {path} belgede korumalı ama token olmadan "
            f"{response.status_code} döndü"
        )
        assert response.json()["detail"]["code"] == "unauthorized"
