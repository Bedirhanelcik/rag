"""POST /agent/ask -- ResearchAgent'in HTTP yuzeyi.

Agent'in kendi davranisi degistirilmez: dongu, karar sinirlari ve status
degerleri oldugu gibi aktarilir. Bu modul yalnizca ceviri ve hata eslemesi
yapar.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
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
from rag_tr.i18n import DEFAULT_LANGUAGE

logger = logging.getLogger(__name__)

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
    request: Request,
    agent=Depends(get_agent),
    agent_factory=Depends(get_agent_factory),
) -> AgentAskResponse:
    # Hic dokuman ingest edilmemisse agent'i hic calistirmayiz: bos korpusta
    # cevap uretilemez ve bosa Gemini cagrisi yapilmis olur. Istemci bunu
    # "sistem bozuk" ile karistirmasin diye ayri bir kod dondurulur.
    service = request.app.state.service
    if service.vector_store.count() == 0:
        raise HTTPException(
            status_code=409,
            detail=_error(
                ErrorCode.CORPUS_EMPTY,
                "Henüz hiç doküman ingest edilmedi, bu yüzden cevap üretilemez.",
            ),
        )

    # Istege ozel top_k ve dil yalnizca ayni LLM/tool'u paylasan hafif bir
    # agent ile uygulanir; RAGService ve embedding modeli asla yeniden
    # kurulmaz. Varsayilan dil ve varsayilan top_k ile gelen istekler
    # singleton agent'i kullanir, yani sicak yol degismedi.
    wants_top_k = payload.top_k is not None and payload.top_k != DEFAULT_TOP_K
    wants_language = payload.language != DEFAULT_LANGUAGE
    if (wants_top_k or wants_language) and agent_factory is not None:
        agent = agent_factory(payload.top_k or DEFAULT_TOP_K, payload.language)

    try:
        # ResearchAgent.run() senkron ve bloklayici (Gemini istekleri + yerel
        # embedding); mevcut /ingest ile ayni threadpool deseni kullanilir.
        result = await run_in_threadpool(agent.run, payload.question)
    except genai_errors.APIError as exc:
        # Upstream hata metni ISTEMCIYE AKTARILMAZ. Gemini'nin 429 govdesi kota
        # metrik adlarini, proje katmanini ve dokumantasyon baglantilarini
        # tasiyan ~1200 karakterlik bir JSON; bunu kullaniciya gostermek hem
        # okunamaz hem de gereksiz altyapi ayrintisi sizdirir. Ayrinti burada
        # loglanir, istemci kisa ve sabit bir cumle alir.
        logger.warning("Gemini API hatası (kod=%s): %s", getattr(exc, "code", None), exc)

        # Yalnizca 429 kota olarak etiketlenir; diger APIError'lar upstream
        # generation hatasidir ve kota gibi gosterilmez.
        if getattr(exc, "code", None) == _QUOTA_STATUS:
            raise HTTPException(
                status_code=_QUOTA_STATUS,
                detail=_error(
                    ErrorCode.QUOTA_EXHAUSTED,
                    "Model sağlayıcısının kullanım kotası şu an dolu. "
                    "Kota yenilendiğinde sorgular yeniden çalışır.",
                ),
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=_error(
                ErrorCode.GENERATION_ERROR,
                "Cevap üretme adımı başarısız oldu. Bu genellikle geçici bir "
                "sorundur; tekrar denemek mantıklı.",
            ),
        ) from exc

    # AgentStatus.TOOL_FAILURE bir istisna degil, agent'in kendi sonucudur:
    # 200 ile dondurulur ki istemci trace'i ve nedeni gorebilsin.
    return _to_response(result)
