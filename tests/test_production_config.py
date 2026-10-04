"""Uretim yapilandirmasi ve guvenlik kontrolleri.

Hicbir Gemini/Anthropic cagrisi yapilmaz, ag erisimi yoktur: servis fake'lenir.
"""

import pathlib

import pytest
from fastapi.testclient import TestClient

from rag_tr.api.main import create_app
from rag_tr.config import Settings
from rag_tr.contracts import ErrorCode
from rag_tr.service import IngestResult


INGEST_TOKEN = "test-ingest-token"
INGEST_HEADERS = {"Authorization": f"Bearer {INGEST_TOKEN}"}


class _Settings:
    embedding_model_name = "fake-embed-model"
    top_k_final = 5
    upload_enabled = False
    upload_max_bytes = 1024
    ingest_api_token = INGEST_TOKEN

    def __init__(self, origins: list[str] | None = None) -> None:
        self._origins = origins or []

    @property
    def allowed_origin_list(self) -> list[str]:
        return list(self._origins)


class _VectorStore:
    def count(self) -> int:
        return 4


class _BM25:
    def size(self) -> int:
        return 4


class StubService:
    def __init__(self, origins: list[str] | None = None, error: Exception | None = None) -> None:
        self.settings = _Settings(origins)
        self.vector_store = _VectorStore()
        self.bm25_index = _BM25()
        self._error = error

    def ingest_files(self, saved_paths) -> IngestResult:
        if self._error is not None:
            raise self._error
        names = [path.name for path in saved_paths]
        return IngestResult(ingested_files=names, failed_files=[], chunk_count=1)


def _client(service: StubService) -> TestClient:
    return TestClient(create_app(service=service), raise_server_exceptions=False)


# --- CORS ---


def test_no_cors_middleware_when_origins_are_unset():
    """Onerilen dagitimda tarayici backend'e dogrudan konusmaz; CORS gerekmez."""
    response = _client(StubService()).get(
        "/health", headers={"Origin": "https://ornek.vercel.app"}
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_allowed_origin_receives_cors_header():
    service = StubService(origins=["https://agent-lab.vercel.app"])

    response = _client(service).get(
        "/health", headers={"Origin": "https://agent-lab.vercel.app"}
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://agent-lab.vercel.app"


def test_disallowed_origin_gets_no_cors_header():
    service = StubService(origins=["https://agent-lab.vercel.app"])

    response = _client(service).get("/health", headers={"Origin": "https://kotu-site.example"})

    # Tarayici, izin basligi olmadigi icin yaniti okuyamaz.
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_rejects_a_disallowed_origin():
    service = StubService(origins=["https://agent-lab.vercel.app"])
    client = _client(service)

    allowed = client.options(
        "/agent/ask",
        headers={
            "Origin": "https://agent-lab.vercel.app",
            "Access-Control-Request-Method": "POST",
        },
    )
    denied = client.options(
        "/agent/ask",
        headers={
            "Origin": "https://kotu-site.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert allowed.headers.get("access-control-allow-origin") == "https://agent-lab.vercel.app"
    assert "access-control-allow-origin" not in denied.headers


def test_cors_never_uses_a_wildcard_or_credentials():
    service = StubService(origins=["https://agent-lab.vercel.app"])

    response = _client(service).get(
        "/health", headers={"Origin": "https://agent-lab.vercel.app"}
    )

    assert response.headers.get("access-control-allow-origin") != "*"
    assert "access-control-allow-credentials" not in response.headers


def test_allowed_origins_parsing_tolerates_whitespace_and_blanks():
    settings = Settings(allowed_origins=" https://a.example , ,https://b.example ")

    assert settings.allowed_origin_list == ["https://a.example", "https://b.example"]


def test_allowed_origins_defaults_to_empty():
    # _env_file=None: yerel .env degil, gercek varsayilan olculuyor.
    assert Settings(_env_file=None).allowed_origin_list == []


# --- /health deployment health check olarak kullanilabilir ---


def test_health_works_without_gemini_and_builds_no_agent():
    service = StubService()
    app = create_app(service=service)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunk_count"] == 4
    assert body["keyword_index_size"] == 4
    # Agent tembel kuruldugu icin /health hicbir model saglayicisina dokunmaz.
    assert app.state.agent is None


def test_health_does_not_require_any_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert _client(StubService()).get("/health").status_code == 200


# --- hata yanitlari sir/iz sizdirmiyor ---


def test_server_errors_do_not_leak_a_traceback():
    service = StubService(error=RuntimeError("chroma down: /secret/path/key=abc123"))

    response = _client(service).post(
        "/ingest", files={"files": ("a.txt", b"veri")}, headers=INGEST_HEADERS
    )

    assert response.status_code == 500
    text = response.text
    assert "Traceback" not in text
    assert "chroma down" not in text, "ic hata mesaji istemciye gitmemeli"
    assert "secret" not in text
    assert "rag_tr" not in text, "kaynak dosya yolu sizmamali"


def test_error_responses_only_carry_a_code_and_a_message():
    response = _client(StubService()).post(
        "/ingest",
        files={"files": ("kotu.exe", b"MZ", "application/octet-stream")},
        headers=INGEST_HEADERS,
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert sorted(detail) == ["code", "message"]
    assert detail["code"] == ErrorCode.UNSUPPORTED_FILE_TYPE.value


# --- uretim varsayilanlari ---


def test_production_defaults_are_safe():
    # Yerel .env uretim varsayilanlarini maskeler; saf varsayilan olculuyor.
    settings = Settings(_env_file=None)

    assert settings.upload_enabled is False, "yukleme dagitimda kapali olmali"
    assert settings.allowed_origin_list == [], "CORS varsayilan olarak kapali"
    assert settings.anthropic_api_key == "", "anlamsiz bir dummy sir gerekmemeli"
    assert settings.chroma_persist_dir, "kalici dizin tanimli olmali"


def test_settings_read_deployment_env_vars(monkeypatch):
    """Platformda verilecek degiskenlerin gercekten okundugunu dogrular."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "/data/chroma")
    monkeypatch.setenv("UPLOAD_ENABLED", "true")
    monkeypatch.setenv("UPLOAD_MAX_BYTES", "2048")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://agent-lab.vercel.app")

    settings = Settings()

    assert settings.chroma_persist_dir == "/data/chroma"
    assert settings.upload_enabled is True
    assert settings.upload_max_bytes == 2048
    assert settings.allowed_origin_list == ["https://agent-lab.vercel.app"]


def test_agent_settings_read_deployment_env_vars(monkeypatch):
    from rag_tr.agent.config import AgentSettings

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-not-used")
    monkeypatch.setenv("GEMINI_AGENT_MODEL", "gemini-2.5-flash-lite")

    settings = AgentSettings()

    assert settings.gemini_api_key == "fake-key-not-used"
    assert settings.gemini_agent_model == "gemini-2.5-flash-lite"


def test_startup_creates_no_upload_dir_when_upload_is_disabled(tmp_path, monkeypatch):
    """Kapali ozellik icin container'da gereksiz yazma denemesi yapilmaz."""
    from rag_tr.api import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("UPLOAD_ENABLED", "false")

    created: list[str] = []
    monkeypatch.setattr(main, "RAGService", lambda settings: created.append("service") or object())

    main.build_service()

    assert (tmp_path / "chroma").exists()
    assert not (tmp_path / "data" / "uploads").exists()


def test_startup_creates_the_upload_dir_when_enabled(tmp_path, monkeypatch):
    from rag_tr.api import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("UPLOAD_ENABLED", "true")
    monkeypatch.setattr(main, "RAGService", lambda settings: object())

    main.build_service()

    assert (tmp_path / "data" / "uploads").exists()


# --- imajda/repoda sir yok ---


def test_dockerignore_excludes_secrets_and_local_state():
    from pathlib import Path

    patterns = Path(".dockerignore").read_text(encoding="utf-8").split()

    for needed in (".env", "data/", ".venv/", ".git/"):
        assert needed in patterns, f"{needed} .dockerignore'da olmali"


def test_dockerfile_uses_the_platform_port_and_no_secrets():
    from pathlib import Path

    dockerfile = Path("Dockerfile.api").read_text(encoding="utf-8")

    assert "${PORT:-8000}" in dockerfile, "platform PORT degiskeni kullanilmali"
    assert "--no-dev" in dockerfile, "uretim imajinda test bagimliliklari olmamali"
    for secret in ("GEMINI_API_KEY=", "ANTHROPIC_API_KEY=", "sk-ant", "AIza"):
        assert secret not in dockerfile, f"Dockerfile'da sir olmamali: {secret}"


@pytest.mark.parametrize("env_file", [".env.example"])
def test_env_example_carries_no_real_secrets(env_file):
    from pathlib import Path

    contents = Path(env_file).read_text(encoding="utf-8")

    assert "sk-ant-api" not in contents
    assert "AIza" not in contents


# --- Anthropic production path'te yok ---


def test_service_module_has_no_module_level_anthropic_import():
    """`anthropic` artik opsiyonel bir bagimlilik; service modulu onu ust
    seviyede import etmemeli, yoksa paket kurulu olmadan uygulama acilamaz."""
    from pathlib import Path

    source = Path("src/rag_tr/service.py").read_text(encoding="utf-8")
    header = source.split("class RAGService", 1)[0]

    assert "import anthropic" not in header, "ust seviyede anthropic import'u olmamali"


def test_constructing_the_service_creates_no_llm_client(tmp_path):
    """Acilista ne Anthropic ne Gemini istemcisi kurulur."""
    from rag_tr.config import Settings
    from rag_tr.retrieval.keyword_search import BM25Index
    from rag_tr.retrieval.vector_store import VectorStore
    from rag_tr.service import RAGService

    settings = Settings(_env_file=None, chroma_persist_dir=str(tmp_path / "chroma"))
    service = RAGService(
        settings,
        embedding_model=object(),
        vector_store=VectorStore(settings.chroma_persist_dir),
        bm25_index=BM25Index(),
    )

    assert service.client is None, "Anthropic istemcisi acilista kurulmamali"


def test_anthropic_is_not_a_production_dependency():
    import re
    from pathlib import Path

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    # Blogun tamami alinmali: "uvicorn[standard]" gibi bir girdi ic ice koseli
    # parantez icerdigi icin basit bir "]" bolmesi blogu erken kesiyor.
    match = re.search(r"^dependencies = \[(.*?)^\]", pyproject, re.S | re.M)
    assert match, "dependencies blogu bulunamadi"
    production_block = match.group(1)
    assert "fastapi" in production_block, "yanlis blok yakalandi"

    assert "anthropic" not in production_block, "anthropic uretim bagimliligi olmamali"
    assert "streamlit" not in production_block, "streamlit uretim bagimliligi olmamali"
    assert "sentence-transformers" not in production_block


# --- embedding modeli yapilandirmasi acilista dogrulanir ---
#
# Gercek bir dagitim hatasindan geliyor: yerel `.env` dosyasi onceki surumden
# kalan `EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small` degerini
# tasiyordu. Uygulama sorunsuz aciliyor, `/health` "ok" donuyor ve modelin adini
# oldugu gibi bildiriyordu; hata ancak ilk embedding cagrisinda, yani ilk
# kullanici sorusunda ortaya cikiyordu. Yapilandirma hatasinin acilista
# gorulmesi gerekir: boylece bozuk bir dagitim trafigi hic almaz.


def test_settings_rejects_a_huggingface_embedding_model_name():
    from pydantic import ValidationError

    from rag_tr.config import Settings

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, embedding_model_name="intfloat/multilingual-e5-small")

    message = str(excinfo.value)
    assert "EMBEDDING_MODEL_NAME" in message
    # Hata, ne yapilmasi gerektigini de soylemeli.
    assert "gemini-embedding-001" in message


def test_settings_rejects_an_empty_embedding_model_name():
    from pydantic import ValidationError

    from rag_tr.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, embedding_model_name="   ")


def test_settings_accepts_a_gemini_embedding_model_name():
    from rag_tr.config import Settings

    settings = Settings(_env_file=None, embedding_model_name="gemini-embedding-001")

    assert settings.embedding_model_name == "gemini-embedding-001"


def test_env_example_declares_a_gemini_embedding_model():
    """Ornek dosya, acilista dogrulamadan gecen bir deger onermeli."""
    from rag_tr.config import Settings

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / ".env.example").read_text(encoding="utf-8")
    line = next(
        row for row in text.splitlines() if row.startswith("EMBEDDING_MODEL_NAME=")
    )
    value = line.split("=", 1)[1].strip()

    assert Settings(_env_file=None, embedding_model_name=value).embedding_model_name == value


# --- test hijyeni ---


def test_tests_never_write_to_the_real_upload_directory(tmp_path):
    """`conftest` izolasyonunun gercekten yururlukte oldugunu dogrular.

    Bu kontrol olmadan bir test, projenin calisma agacindaki `data/uploads`
    dizinine dosya birakabilir; daha once tam olarak bu oluyordu."""
    from pathlib import Path

    from rag_tr.api import routes

    assert routes.UPLOAD_DIR != Path("data/uploads"), (
        "testler gerçek yükleme dizinine yazmamalı"
    )
    assert str(tmp_path) in str(routes.UPLOAD_DIR)
