"""Agent erisim katmani.

HTTP katmani agent'in nasil kuruldugunu bilmez: yalnizca `get_agent`'i cagirir.
Agent app.state uzerinde singleton olarak tutulur -- RAGService (ve onunla
birlikte embedding modeli) istek basina yeniden kurulmaz.
"""

import threading

from fastapi import HTTPException, Request

from rag_tr.agent.config import AgentSettings
from rag_tr.agent.gemini import GeminiAgentLLM
from rag_tr.agent.loop import ResearchAgent
from rag_tr.agent.tools import RagSearchTool
from rag_tr.contracts import ErrorCode
from rag_tr.i18n import DEFAULT_LANGUAGE

_build_lock = threading.Lock()


def _unavailable(message: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={"code": ErrorCode.AGENT_UNAVAILABLE.value, "message": message},
    )


def get_agent(request: Request) -> ResearchAgent:
    """app.state'teki agent'i dondurur; yoksa bir kez kurup onbellege alir.

    Tembel kurulum sayesinde create_app() Gemini key'i olmadan da calisir
    (RAG-only dagitim ve testler); key yoksa yalnizca bu uc nokta 503 verir."""
    app = request.app
    agent = getattr(app.state, "agent", None)
    if agent is not None:
        return agent

    with _build_lock:
        agent = getattr(app.state, "agent", None)
        if agent is None:
            agent, factory = _build_agent(app)
            app.state.agent = agent
            app.state.agent_factory = factory
    return agent


def get_agent_factory(request: Request):
    """Istege ozel top_k ve dil icin, ayni LLM ve arama tool'unu paylasan
    bir agent uretir. Enjekte edilmis (fake) agent'larda bulunmaz, None
    doner."""
    return getattr(request.app.state, "agent_factory", None)


def _build_agent(app):
    settings = AgentSettings()
    if not settings.gemini_api_key:
        raise _unavailable(
            "GEMINI_API_KEY tanımlı değil; agent uç noktası devre dışı. "
            "Sunucu tarafı ortam değişkenlerine ekleyin."
        )

    service = getattr(app.state, "service", None)
    if service is None:
        raise _unavailable("RAG servisi kurulu değil.")

    llm = GeminiAgentLLM.from_env(settings)
    tool = RagSearchTool(service)

    def factory(top_k: int, language: str = DEFAULT_LANGUAGE) -> ResearchAgent:
        # RAGService, embedding modeli ve Gemini client PAYLASILIR: istek
        # basina yeniden kurulan sey yalnizca ince bir dongu nesnesi ve
        # system prompt'una dil direktifi eklenmis bir LLM sarmalayicisi.
        # Dolayisiyla dil destegi ek bir model cagrisi uretmez.
        return ResearchAgent(
            llm=llm.for_language(language),
            search_tool=tool,
            top_k=top_k,
            language=language,
        )

    return ResearchAgent(llm=llm, search_tool=tool), factory
