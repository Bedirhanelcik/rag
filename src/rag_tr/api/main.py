from pathlib import Path

from fastapi import FastAPI

from rag_tr.api.routes import router
from rag_tr.config import Settings
from rag_tr.service import RAGService


def create_app() -> FastAPI:
    app = FastAPI(title="Türkçe RAG API")
    settings = Settings()
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    Path("data/uploads").mkdir(parents=True, exist_ok=True)
    app.state.service = RAGService(settings)
    app.include_router(router)
    return app


app = create_app()
