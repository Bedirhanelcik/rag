"""Agent'in karar dongusu.

Kasitli olarak kucuk bir durum makinesi: framework yok, gizli sihir yok. Akis
sabit bir pipeline degil -- her adimda bir karar var ve bu kararlar kosuyu
degistirir:

    ASSESS   -> arama gerekli mi?            (gerekmiyorsa hic tool cagrilmaz)
    SEARCH   -> RAG'dan pasaj getir
    INSPECT  -> pasajlar yetiyor mu?
    REFINE   -> yetmiyorsa sorguyu degistirip tekrar ara (sinirli)
    ANSWER   -> yalnizca yeterli baglam varsa cevap uret
"""

from rag_tr.agent.contracts import (
    AgentAction,
    AgentResult,
    AgentStatus,
    PassageAssessment,
    RetrievalDecision,
    Step,
    ToolCall,
    ToolStatus,
)
from rag_tr.agent.tools import SearchTool
from rag_tr.contracts import NO_CONTEXT_MESSAGE, Passage

DEFAULT_TOP_K = 5
DEFAULT_MAX_SEARCHES = 2
_MAX_DETAIL_CHARS = 200


class AgentLLM:
    """Agent'in ihtiyac duydugu LLM yuzeyi (Protocol gibi kullanilir).

    Dort ayri karar/uretim noktasi; her biri tiplenmis bir sonuc dondurur ve
    hicbiri serbest dusunce zinciri disa vermez."""

    def needs_retrieval(self, question: str) -> RetrievalDecision: ...

    def assess_passages(self, question: str, passages: list[Passage]) -> PassageAssessment: ...

    def synthesize(self, question: str, passages: list[Passage]) -> str: ...

    def answer_directly(self, question: str) -> str: ...


def _trim(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= _MAX_DETAIL_CHARS else text[: _MAX_DETAIL_CHARS - 1] + "…"


def _normalize(query: str) -> str:
    return " ".join(query.split()).casefold()


class ResearchAgent:
    def __init__(
        self,
        llm: AgentLLM,
        search_tool: SearchTool,
        *,
        top_k: int = DEFAULT_TOP_K,
        max_searches: int = DEFAULT_MAX_SEARCHES,
    ) -> None:
        if max_searches < 1:
            raise ValueError("max_searches en az 1 olmali")
        self._llm = llm
        self._tool = search_tool
        self._top_k = top_k
        self._max_searches = max_searches

    def run(self, question: str) -> AgentResult:
        steps: list[Step] = []
        tool_calls: list[ToolCall] = []

        decision = self._llm.needs_retrieval(question)
        steps.append(
            Step(
                action=AgentAction.ASSESSED_QUESTION,
                detail=_trim(decision.reason)
                or ("arama gerekli" if decision.needs_retrieval else "arama gerekmiyor"),
            )
        )

        if not decision.needs_retrieval:
            steps.append(
                Step(
                    action=AgentAction.ANSWERED_WITHOUT_RETRIEVAL,
                    detail="dokuman aramasi yapilmadan cevaplandi",
                )
            )
            return AgentResult(
                answer=self._llm.answer_directly(question),
                status=AgentStatus.ANSWERED_WITHOUT_RETRIEVAL,
                sources=[],
                steps=steps,
                tool_calls=tool_calls,
            )

        query = (decision.search_query or question).strip() or question
        tried: set[str] = set()
        grounded: list[Passage] = []

        for attempt in range(1, self._max_searches + 1):
            tried.add(_normalize(query))
            outcome = self._tool.search(query, top_k=self._top_k)
            tool_calls.append(
                ToolCall(
                    tool=self._tool.name,
                    query=query,
                    top_k=self._top_k,
                    source_file=None,
                    status=outcome.status,
                    passages_found=len(outcome.passages),
                )
            )
            steps.append(
                Step(
                    action=AgentAction.SEARCHED_RAG,
                    detail=f"deneme {attempt}: {len(outcome.passages)} pasaj bulundu",
                )
            )

            if outcome.status is ToolStatus.FAILED:
                # Retrieval katmani hatasi: RAG sozlesmesine gore tekrar denemek
                # anlamli degil, dongu burada kesilir.
                detail = _trim(f"retrieval tool hatasi: {outcome.error}")
                steps.append(Step(action=AgentAction.TOOL_FAILED, detail=detail))
                return AgentResult(
                    answer="Belge aramasi sirasinda bir hata olustu, bu nedenle cevap üretilemedi.",
                    status=AgentStatus.TOOL_FAILURE,
                    sources=[],
                    steps=steps,
                    tool_calls=tool_calls,
                )

            assessment = self._llm.assess_passages(question, outcome.passages)
            if assessment.sufficient and outcome.passages:
                grounded = outcome.passages
                break

            if attempt == self._max_searches:
                break

            refined = (assessment.refined_query or "").strip()
            if not refined or _normalize(refined) in tried:
                # Yeni bir fikir yoksa veya ayni sorgu tekrar ediliyorsa bos
                # yere tool cagirmak yerine dur.
                break

            query = refined
            steps.append(
                Step(
                    action=AgentAction.REFINED_QUERY,
                    detail=_trim(assessment.reason) or "sorgu yeniden formullendi",
                )
            )

        if grounded:
            answer = self._llm.synthesize(question, grounded)
            steps.append(
                Step(
                    action=AgentAction.ANSWERED_FROM_CONTEXT,
                    detail=f"{len(grounded)} pasaj temel alinarak cevaplandi",
                )
            )
            return AgentResult(
                answer=answer,
                status=AgentStatus.ANSWERED,
                sources=list(grounded),
                steps=steps,
                tool_calls=tool_calls,
            )

        steps.append(
            Step(
                action=AgentAction.DECLARED_INSUFFICIENT_CONTEXT,
                detail=f"{len(tool_calls)} arama sonrasi yeterli baglam bulunamadi",
            )
        )
        return AgentResult(
            answer=NO_CONTEXT_MESSAGE,
            status=AgentStatus.INSUFFICIENT_CONTEXT,
            sources=[],
            steps=steps,
            tool_calls=tool_calls,
        )
