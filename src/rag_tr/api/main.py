from pathlib import Path
from typing import Any

from fastapi import FastAPI

from rag_tr.api.agent_routes import router as agent_router
from rag_tr.api.routes import router
from rag_tr.config import Settings
from rag_tr.service import RAGService


def build_service() -> RAGService:
    """Uretim servisini kurar: .env okur, kalici dizinleri olusturur, gercek
    embedding modelini ve Anthropic client'ini yaratir."""
    settings = Settings()
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    Path("data/uploads").mkdir(parents=True, exist_ok=True)
    return RAGService(settings)


def create_app(service: RAGService | None = None, agent=None) -> FastAPI:
    """`agent` verilirse app.state'e yerlestirilir (testler fake enjekte eder);
    verilmezse ilk /agent/ask isteginde tembel olarak kurulur, boylece RAG-only
    dagitim Gemini key'i olmadan da calismaya devam eder."""
    app = FastAPI(title="Türkçe RAG API")
    app.state.service = service if service is not None else build_service()
    app.state.agent = agent
    app.state.agent_factory = None
    app.include_router(router)
    app.include_router(agent_router)
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
