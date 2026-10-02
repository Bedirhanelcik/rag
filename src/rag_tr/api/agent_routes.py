"""POST /agent/ask -- ResearchAgent'in HTTP yuzeyi.

Agent'in kendi davranisi degistirilmez: dongu, karar sinirlari ve status
degerleri oldugu gibi aktarilir. Bu modul yalnizca ceviri ve hata eslemesi
yapar.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from google.genai import errors as genai_errors

from rag_tr.agent.contracts import AgentResult
from rag_tr.agent.loop import DEFAULT_TOP_K
from rag_tr.api.agent_schemas import (
    AgentAskRequest,
    AgentAskResponse,
    AgentSourceItem,
    AgentStepItem,
    AgentToolCallItem,
)
from rag_tr.api.deps import get_agent, get_agent_factory
from rag_tr.contracts import ErrorCode

router = APIRouter(prefix="/agent", tags=["agent"])

_QUOTA_STATUS = 429


def _error(code: ErrorCode, message: str) -> dict:
    return {"code": code.value, "message": message}


def _to_response(result: AgentResult) -> AgentAskResponse:
    return AgentAskResponse(
        answer=result.answer,
        status=result.status,
        sources=[
            AgentSourceItem(
                chunk_id=source.chunk_id,
                source_file=source.source_file,
                page_number=source.page_number,
                text=source.text,
                rank=source.rank,
                score=source.score,
            )
            for source in result.sources
        ],
        steps=[AgentStepItem(action=step.action, detail=step.detail) for step in result.steps],
        tool_calls=[
            AgentToolCallItem(
                tool=call.tool,
                query=call.query,
                top_k=call.top_k,
                source_file=call.source_file,
                status=call.status,
                passages_found=call.passages_found,
            )
            for call in result.tool_calls
        ],
    )


@router.post("/ask", response_model=AgentAskResponse)
async def ask(
    payload: AgentAskRequest,
    agent=Depends(get_agent),
    agent_factory=Depends(get_agent_factory),
) -> AgentAskResponse:
    # Istege ozel top_k yalnizca ayni LLM/tool'u paylasan hafif bir agent ile
    # uygulanir; RAGService ve embedding modeli asla yeniden kurulmaz.
    if payload.top_k is not None and payload.top_k != DEFAULT_TOP_K and agent_factory is not None:
        agent = agent_factory(payload.top_k)

    try:
        # ResearchAgent.run() senkron ve bloklayici (Gemini istekleri + yerel
        # embedding); mevcut /ingest ile ayni threadpool deseni kullanilir.
        result = await run_in_threadpool(agent.run, payload.question)
    except genai_errors.APIError as exc:
        # Yalnizca 429 kota olarak etiketlenir; diger APIError'lar upstream
        # generation hatasidir ve kota gibi gosterilmez.
        if getattr(exc, "code", None) == _QUOTA_STATUS:
            raise HTTPException(
                status_code=_QUOTA_STATUS,
                detail=_error(
                    ErrorCode.QUOTA_EXHAUSTED,
                    f"Gemini kotası tükendi, lütfen daha sonra tekrar deneyin: {exc}",
                ),
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=_error(ErrorCode.GENERATION_ERROR, f"Gemini API hatası: {exc}"),
        ) from exc

    # AgentStatus.TOOL_FAILURE bir istisna degil, agent'in kendi sonucudur:
    # 200 ile dondurulur ki istemci trace'i ve nedeni gorebilsin.
    return _to_response(result)
