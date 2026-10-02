"""ResearchAgent karar dongusu testleri.

Hicbir gercek LLM veya API kullanilmaz: LLM ve arama tool'u scripted fake'lerle
degistirilir, boylece dongu tamamen deterministik olarak dogrulanir.
"""

from rag_tr.agent.contracts import (
    AgentAction,
    AgentStatus,
    PassageAssessment,
    RetrievalDecision,
    SearchOutcome,
    ToolStatus,
)
from rag_tr.agent.loop import ResearchAgent
from rag_tr.contracts import NO_CONTEXT_MESSAGE, Passage


def _passage(chunk_id="cografya.md::0", text="Türkiye'nin başkenti Ankara'dır.", rank=1) -> Passage:
    return Passage(
        chunk_id=chunk_id,
        source_file=chunk_id.split("::")[0],
        page_number=None,
        text=text,
        rank=rank,
        score=1.0 / rank,
    )


def _found(*passages) -> SearchOutcome:
    return SearchOutcome(status=ToolStatus.OK, passages=list(passages))


def _empty() -> SearchOutcome:
    return SearchOutcome(status=ToolStatus.EMPTY, passages=[])


def _failed(error="chroma down") -> SearchOutcome:
    return SearchOutcome(status=ToolStatus.FAILED, passages=[], error=error)


class ScriptedLLM:
    """Kararlari onceden belirlenmis, cagrilari kaydeden fake LLM."""

    def __init__(
        self,
        *,
        decision: RetrievalDecision,
        assessments: list[PassageAssessment] | None = None,
        answer: str = "Ankara'dır [cografya.md::0].",
        direct_answer: str = "Merhaba, nasıl yardımcı olabilirim?",
    ) -> None:
        self.decision = decision
        self.assessments = list(assessments or [])
        self.answer = answer
        self.direct_answer = direct_answer
        self.calls: list[str] = []
        self.synthesized_with: list[list[Passage]] = []

    def needs_retrieval(self, question: str) -> RetrievalDecision:
        self.calls.append("needs_retrieval")
        return self.decision

    def assess_passages(self, question: str, passages: list[Passage]) -> PassageAssessment:
        self.calls.append("assess_passages")
        if self.assessments:
            return self.assessments.pop(0)
        return PassageAssessment(sufficient=False, reason="yetersiz")

    def synthesize(self, question: str, passages: list[Passage]) -> str:
        self.calls.append("synthesize")
        self.synthesized_with.append(list(passages))
        return self.answer

    def answer_directly(self, question: str) -> str:
        self.calls.append("answer_directly")
        return self.direct_answer


class ScriptedSearchTool:
    name = "rag_search"

    def __init__(self, outcomes: list[SearchOutcome]) -> None:
        self.outcomes = list(outcomes)
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 5, source_file: str | None = None) -> SearchOutcome:
        self.queries.append(query)
        return self.outcomes.pop(0) if self.outcomes else _empty()


def _actions(result) -> list[AgentAction]:
    return [step.action for step in result.steps]


# --- 1. retrieval gereksizse dogrudan cevap ---


def test_direct_answer_path_skips_retrieval_entirely():
    llm = ScriptedLLM(decision=RetrievalDecision(needs_retrieval=False, reason="selamlama"))
    tool = ScriptedSearchTool([])
    agent = ResearchAgent(llm=llm, search_tool=tool)

    result = agent.run("Merhaba!")

    assert result.status is AgentStatus.ANSWERED_WITHOUT_RETRIEVAL
    assert result.answer == "Merhaba, nasıl yardımcı olabilirim?"
    assert result.sources == []
    assert result.tool_calls == []
    assert tool.queries == [], "retrieval gerekmiyorken tool cagrilmamali"
    assert llm.calls == ["needs_retrieval", "answer_directly"]
    assert _actions(result) == [
        AgentAction.ASSESSED_QUESTION,
        AgentAction.ANSWERED_WITHOUT_RETRIEVAL,
    ]


# --- 2. RAG arama yolu ---


def test_rag_search_path_produces_grounded_answer():
    llm = ScriptedLLM(
        decision=RetrievalDecision(needs_retrieval=True, reason="dokuman bilgisi gerekiyor"),
        assessments=[PassageAssessment(sufficient=True, reason="yeterli")],
    )
    tool = ScriptedSearchTool([_found(_passage())])
    agent = ResearchAgent(llm=llm, search_tool=tool)

    result = agent.run("Türkiye'nin başkenti neresidir?")

    assert result.status is AgentStatus.ANSWERED
    assert result.answer == "Ankara'dır [cografya.md::0]."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool == "rag_search"
    assert result.tool_calls[0].passages_found == 1
    assert _actions(result) == [
        AgentAction.ASSESSED_QUESTION,
        AgentAction.SEARCHED_RAG,
        AgentAction.ANSWERED_FROM_CONTEXT,
    ]


def test_agent_uses_the_llms_search_query_when_it_differs_from_the_question():
    llm = ScriptedLLM(
        decision=RetrievalDecision(
            needs_retrieval=True, search_query="Türkiye başkent", reason="anahtar kelimeler"
        ),
        assessments=[PassageAssessment(sufficient=True)],
    )
    tool = ScriptedSearchTool([_found(_passage())])

    ResearchAgent(llm=llm, search_tool=tool).run("Acaba Türkiye'nin başkenti neresidir?")

    assert tool.queries == ["Türkiye başkent"]


# --- 3. yetersiz baglam yolu ---


def test_insufficient_context_is_reported_explicitly():
    llm = ScriptedLLM(
        decision=RetrievalDecision(needs_retrieval=True),
        assessments=[PassageAssessment(sufficient=False), PassageAssessment(sufficient=False)],
    )
    tool = ScriptedSearchTool([_empty(), _empty()])
    agent = ResearchAgent(llm=llm, search_tool=tool)

    result = agent.run("Mars'ta hava nasıl?")

    assert result.status is AgentStatus.INSUFFICIENT_CONTEXT
    assert result.answer == NO_CONTEXT_MESSAGE
    assert result.sources == []
    assert "synthesize" not in llm.calls, "baglam yoksa cevap uretilmemeli"
    assert AgentAction.DECLARED_INSUFFICIENT_CONTEXT in _actions(result)


def test_passages_found_but_judged_insufficient_does_not_fabricate_an_answer():
    llm = ScriptedLLM(
        decision=RetrievalDecision(needs_retrieval=True),
        assessments=[PassageAssessment(sufficient=False), PassageAssessment(sufficient=False)],
    )
    tool = ScriptedSearchTool([_found(_passage()), _found(_passage())])

    result = ResearchAgent(llm=llm, search_tool=tool).run("Soru?")

    assert result.status is AgentStatus.INSUFFICIENT_CONTEXT
    assert result.sources == []
    assert "synthesize" not in llm.calls


# --- 4. sorgu iyilestirme / ikinci retrieval ---


def test_refines_the_query_and_succeeds_on_the_second_attempt():
    llm = ScriptedLLM(
        decision=RetrievalDecision(needs_retrieval=True),
        assessments=[
            PassageAssessment(sufficient=False, refined_query="Ankara başkent", reason="çok genel"),
            PassageAssessment(sufficient=True, reason="yeterli"),
        ],
    )
    tool = ScriptedSearchTool([_empty(), _found(_passage())])
    agent = ResearchAgent(llm=llm, search_tool=tool)

    result = agent.run("başkent")

    assert result.status is AgentStatus.ANSWERED
    assert tool.queries == ["başkent", "Ankara başkent"]
    assert len(result.tool_calls) == 2
    assert _actions(result) == [
        AgentAction.ASSESSED_QUESTION,
        AgentAction.SEARCHED_RAG,
        AgentAction.REFINED_QUERY,
        AgentAction.SEARCHED_RAG,
        AgentAction.ANSWERED_FROM_CONTEXT,
    ]


def test_search_attempts_are_capped():
    llm = ScriptedLLM(
        decision=RetrievalDecision(needs_retrieval=True),
        assessments=[
            PassageAssessment(sufficient=False, refined_query="ikinci"),
            PassageAssessment(sufficient=False, refined_query="ucuncu"),
            PassageAssessment(sufficient=False, refined_query="dorduncu"),
        ],
    )
    tool = ScriptedSearchTool([_empty(), _empty(), _empty()])
    agent = ResearchAgent(llm=llm, search_tool=tool, max_searches=2)

    result = agent.run("ilk")

    assert len(result.tool_calls) == 2
    assert tool.queries == ["ilk", "ikinci"]
    assert result.status is AgentStatus.INSUFFICIENT_CONTEXT


def test_does_not_repeat_an_identical_query():
    llm = ScriptedLLM(
        decision=RetrievalDecision(needs_retrieval=True),
        assessments=[PassageAssessment(sufficient=False, refined_query="  Başkent  ")],
    )
    tool = ScriptedSearchTool([_empty(), _empty()])
    agent = ResearchAgent(llm=llm, search_tool=tool, max_searches=3)

    result = agent.run("başkent")

    assert tool.queries == ["başkent"], "ayni sorgu tekrar calistirilmamali"
    assert result.status is AgentStatus.INSUFFICIENT_CONTEXT


def test_missing_refined_query_stops_the_loop_early():
    llm = ScriptedLLM(
        decision=RetrievalDecision(needs_retrieval=True),
        assessments=[PassageAssessment(sufficient=False, refined_query=None)],
    )
    tool = ScriptedSearchTool([_empty(), _empty()])
    agent = ResearchAgent(llm=llm, search_tool=tool, max_searches=3)

    result = agent.run("soru")

    assert len(result.tool_calls) == 1
    assert result.status is AgentStatus.INSUFFICIENT_CONTEXT


# --- 5. kaynak aktarimi ---


def test_sources_are_propagated_from_the_passages_actually_used():
    used = [_passage("cografya.md::0", rank=1), _passage("tarih.md::2", text="1299", rank=2)]
    llm = ScriptedLLM(
        decision=RetrievalDecision(needs_retrieval=True),
        assessments=[PassageAssessment(sufficient=True)],
    )
    tool = ScriptedSearchTool([_found(*used)])

    result = ResearchAgent(llm=llm, search_tool=tool).run("Soru?")

    assert [s.chunk_id for s in result.sources] == ["cografya.md::0", "tarih.md::2"]
    assert [s.source_file for s in result.sources] == ["cografya.md", "tarih.md"]
    assert llm.synthesized_with[0] == used, "sentez tam olarak dondurulen kaynaklari gormeli"


def test_sources_come_from_the_successful_attempt_not_the_discarded_one():
    stale = _passage("eski.md::0", text="alakasiz", rank=1)
    fresh = _passage("cografya.md::0", rank=1)
    llm = ScriptedLLM(
        decision=RetrievalDecision(needs_retrieval=True),
        assessments=[
            PassageAssessment(sufficient=False, refined_query="daha iyi"),
            PassageAssessment(sufficient=True),
        ],
    )
    tool = ScriptedSearchTool([_found(stale), _found(fresh)])

    result = ResearchAgent(llm=llm, search_tool=tool).run("soru")

    assert [s.chunk_id for s in result.sources] == ["cografya.md::0"]


# --- 6. tool hatasi ---


def test_tool_failure_is_surfaced_and_stops_the_loop():
    llm = ScriptedLLM(decision=RetrievalDecision(needs_retrieval=True))
    tool = ScriptedSearchTool([_failed("chroma down")])
    agent = ResearchAgent(llm=llm, search_tool=tool)

    result = agent.run("Soru?")

    assert result.status is AgentStatus.TOOL_FAILURE
    assert result.sources == []
    assert "synthesize" not in llm.calls
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].status is ToolStatus.FAILED
    assert AgentAction.TOOL_FAILED in _actions(result)
    assert "chroma down" in result.answer or "chroma down" in result.steps[-1].detail


def test_tool_failure_is_not_retried():
    llm = ScriptedLLM(decision=RetrievalDecision(needs_retrieval=True))
    tool = ScriptedSearchTool([_failed(), _found(_passage())])
    agent = ResearchAgent(llm=llm, search_tool=tool, max_searches=3)

    result = ResearchAgent(llm=llm, search_tool=tool, max_searches=3).run("Soru?")

    assert len(result.tool_calls) == 1
    assert result.status is AgentStatus.TOOL_FAILURE


# --- 7. determinizm ve trace guvenligi ---


def test_identical_inputs_produce_identical_results():
    def _build():
        llm = ScriptedLLM(
            decision=RetrievalDecision(needs_retrieval=True),
            assessments=[
                PassageAssessment(sufficient=False, refined_query="ikinci"),
                PassageAssessment(sufficient=True),
            ],
        )
        tool = ScriptedSearchTool([_empty(), _found(_passage())])
        return ResearchAgent(llm=llm, search_tool=tool)

    first = _build().run("soru")
    second = _build().run("soru")

    assert first == second


def test_trace_contains_only_safe_high_level_actions():
    llm = ScriptedLLM(
        decision=RetrievalDecision(needs_retrieval=True, reason="dokuman gerekiyor"),
        assessments=[PassageAssessment(sufficient=True, reason="yeterli")],
    )
    tool = ScriptedSearchTool([_found(_passage())])

    result = ResearchAgent(llm=llm, search_tool=tool).run("Soru?")

    for step in result.steps:
        assert isinstance(step.action, AgentAction)
        assert len(step.detail) <= 200, "trace detayi kisa ve ozet kalmali"
