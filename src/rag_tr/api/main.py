from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_tr.api.agent_routes import router as agent_router
from rag_tr.api.document_routes import router as document_router
from rag_tr.api.errors import register_error_handlers
from rag_tr.api.openapi import build_openapi
from rag_tr.api.routes import router
from rag_tr.api.upload_routes import router as upload_router
from rag_tr.config import Settings
from rag_tr.service import RAGService


def build_service() -> RAGService:
    """Uretim servisini kurar: .env okur, kalici dizinleri olusturur, gercek
    embedding modelini ve Anthropic client'ini yaratir."""
    settings = Settings()
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    # Yukleme dizini yalnizca ozellik acikken olusturulur: kapali bir dagitimda
    # (varsayilan) container'da gereksiz bir yazma denemesi yapilmaz.
    if settings.upload_enabled:
        Path("data/uploads").mkdir(parents=True, exist_ok=True)
    return RAGService(settings)


def _configure_cors(app: FastAPI) -> None:
    """ALLOWED_ORIGINS tanimliysa CORS'u YALNIZCA o origin'lere acar.

    Bos birakilirsa middleware hic eklenmez; onerilen dagitimda tarayici
    backend'e dogrudan konusmadigi (Next.js sunucu tarafi proxy'ledigi) icin
    CORS'a ihtiyac yoktur. Joker (*) hicbir durumda kullanilmaz ve kimlik
    bilgisi tasinmasina izin verilmez.

    `getattr` kullaniliyor: testlerde enjekte edilen sahte servislerin ayar
    nesnesinde bu alan bulunmayabilir.
    """
    settings = getattr(app.state.service, "settings", None)
    origins = getattr(settings, "allowed_origin_list", None) or []
    if not origins:
        return

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "accept"],
    )

def create_app(service: RAGService | None = None, agent=None) -> FastAPI:
    """`agent` verilirse app.state'e yerlestirilir (testler fake enjekte eder);
    verilmezse ilk /agent/ask isteginde tembel olarak kurulur, boylece RAG-only
    dagitim Gemini key'i olmadan da calismaya devam eder."""
    app = FastAPI(title="Türkçe RAG API")
    app.state.service = service if service is not None else build_service()
    _configure_cors(app)
    register_error_handlers(app)
    app.state.agent = agent
    app.state.agent_factory = None
    app.include_router(router)
    app.include_router(agent_router)
    app.include_router(upload_router)
    app.include_router(document_router)
    # Dosya alanlarinin Swagger UI'da "Choose File" olarak cikmasi icin:
    # ayrinti icin rag_tr.api.openapi.
    app.openapi = lambda: build_openapi(app)
    return app


def __getattr__(name: str) -> Any:
    """`uvicorn rag_tr.api.main:app` komutunu korur ama uygulamayi yalnizca
    gercekten istendiginde kurar; modulu import etmek (ornegin testlerde) artik
    .env, embedding model indirmesi veya API key gerektirmiyor."""
    if name == "app":
        app = create_app()
        globals()["app"] = app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
