"""Agent degerlendirme adapter'i testleri.

Hicbir gercek API cagrisi yapilmaz: agent, LLM ve judge client fake'lenir.
"""

import asyncio
from pathlib import Path

from promptevals.models import AssertionResult

from rag_tr.agent.contracts import (
    AgentAction,
    AgentResult,
    AgentStatus,
    Step,
    ToolCall,
    ToolStatus,
)
from rag_tr.agent.eval import (
    AgentCase,
    JudgeSpec,
    compile_assertions,
    evaluate_agent_case,
    load_suite,
    render_observation,
    run_agent_eval,
)
from rag_tr.contracts import NO_CONTEXT_MESSAGE, Passage

SUITE_PATH = Path(__file__).resolve().parent.parent / "eval" / "agent_suite.yaml"


def _passage(chunk_id="turkiye_cografyasi.md::0", rank=1):
    return Passage(
        chunk_id=chunk_id,
        source_file=chunk_id.split("::")[0],
        page_number=None,
        text="Ağrı Dağı 5137 metredir.",
        rank=rank,
        score=1.0 / rank,
    )


def _tool_call(query="Türkiye en yüksek dağ", found=1):
    return ToolCall(
        tool="rag_search",
        query=query,
        top_k=5,
        source_file=None,
        status=ToolStatus.OK,
        passages_found=found,
    )


def _answered_result(answer="Ağrı Dağı, 5137 metre [turkiye_cografyasi.md::0]."):
    return AgentResult(
        answer=answer,
        status=AgentStatus.ANSWERED,
        sources=[_passage()],
        steps=[
            Step(AgentAction.ASSESSED_QUESTION, "olgusal"),
            Step(AgentAction.SEARCHED_RAG, "deneme 1: 1 pasaj bulundu"),
            Step(AgentAction.ANSWERED_FROM_CONTEXT, "1 pasaj"),
        ],
        tool_calls=[_tool_call()],
    )


def _insufficient_result():
    return AgentResult(
        answer=NO_CONTEXT_MESSAGE,
        status=AgentStatus.INSUFFICIENT_CONTEXT,
        sources=[],
        steps=[
            Step(AgentAction.ASSESSED_QUESTION, ""),
            Step(AgentAction.SEARCHED_RAG, "deneme 1: 0 pasaj bulundu"),
            Step(AgentAction.DECLARED_INSUFFICIENT_CONTEXT, "yeterli baglam yok"),
        ],
        tool_calls=[_tool_call(query="Japonya başkent", found=0)],
    )


def _direct_result():
    return AgentResult(
        answer="İyiyim, teşekkürler!",
        status=AgentStatus.ANSWERED_WITHOUT_RETRIEVAL,
        sources=[],
        steps=[
            Step(AgentAction.ASSESSED_QUESTION, "selamlama"),
            Step(AgentAction.ANSWERED_WITHOUT_RETRIEVAL, ""),
        ],
        tool_calls=[],
    )


def _refined_result():
    return AgentResult(
        answer="Yedi bölge [turkiye_cografyasi.md::0].",
        status=AgentStatus.ANSWERED,
        sources=[_passage()],
        steps=[
            Step(AgentAction.ASSESSED_QUESTION, ""),
            Step(AgentAction.SEARCHED_RAG, "deneme 1: 0 pasaj bulundu"),
            Step(AgentAction.REFINED_QUERY, "çok genel"),
            Step(AgentAction.SEARCHED_RAG, "deneme 2: 1 pasaj bulundu"),
            Step(AgentAction.ANSWERED_FROM_CONTEXT, ""),
        ],
        tool_calls=[_tool_call(query="bölge", found=0), _tool_call(query="coğrafi bölge", found=1)],
    )


class FakeAgent:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.questions: list[str] = []

    def run(self, question: str) -> AgentResult:
        self.questions.append(question)
        if self._error is not None:
            raise self._error
        return self._result


class FakeJudgeClient:
    """run_llm_judge'in Gemini yolunu taklit eder."""

    def __init__(self, score=5, reasoning="iyi") -> None:
        self.score = score
        self.reasoning = reasoning
        self.calls = 0

    class _Verdict:
        def __init__(self, score, reasoning):
            self.score = score
            self.reasoning = reasoning

    @property
    def aio(self):
        return self

    @property
    def models(self):
        return self

    async def generate_content(self, **kwargs):
        self.calls += 1

        class _Response:
            parsed = None

        response = _Response()
        response.parsed = self._Verdict(self.score, self.reasoning)
        return response


def _run(coro):
    return asyncio.run(coro)


# --- gozlem belgesi ---


def test_observation_contains_every_observable_field():
    observation = render_observation(_answered_result())

    assert "status: [answered]" in observation
    assert "tool_calls: 1" in observation
    assert "tools_used: [rag_search]" in observation
    assert "search_queries: [Türkiye en yüksek dağ]" in observation
    assert "[searched_rag]" in observation
    assert "sources: [turkiye_cografyasi.md::0]" in observation
    assert "source_files: [turkiye_cografyasi.md]" in observation
    assert "answer: Ağrı Dağı" in observation


def test_observation_distinguishes_answered_from_answered_without_retrieval():
    """'answered' metni digerinin on eki -- parantezler bu karismayi onler."""
    answered = render_observation(_answered_result())
    direct = render_observation(_direct_result())

    assert "status: [answered]" in answered
    assert "status: [answered]" not in direct
    assert "status: [answered_without_retrieval]" in direct


def test_observation_collapses_multiline_answers_to_one_line():
    observation = render_observation(_answered_result(answer="birinci satır\nikinci satır"))

    assert "answer: birinci satır ikinci satır" in observation


def test_observation_is_deterministic():
    assert render_observation(_answered_result()) == render_observation(_answered_result())


# --- 1. olgusal RAG sorusu ---


def test_factual_rag_case_passes():
    case = AgentCase(
        id="factual",
        question="Türkiye'nin en yüksek dağı hangisidir?",
        expect_status="answered",
        expect_tool="rag_search",
        expect_action=["searched_rag", "answered_from_context"],
        expect_source="turkiye_cografyasi.md::0",
        answer_contains=["Ağrı"],
        answer_not_contains=[NO_CONTEXT_MESSAGE],
    )
    agent = FakeAgent(_answered_result())

    result = _run(evaluate_agent_case(agent, case))

    assert result.passed, [a.detail for a in result.assertion_results if not a.passed]
    assert agent.questions == ["Türkiye'nin en yüksek dağı hangisidir?"]
    assert len(result.assertion_results) == 7


# --- 2. korpus disi: uydurma yok ---


def test_insufficient_context_case_passes():
    case = AgentCase(
        id="outside",
        question="Japonya'nın başkenti neresidir?",
        expect_status="insufficient_context",
        expect_action="declared_insufficient_context",
        answer_contains=[NO_CONTEXT_MESSAGE],
        answer_not_contains=["Tokyo"],
    )

    result = _run(evaluate_agent_case(FakeAgent(_insufficient_result()), case))

    assert result.passed


def test_hallucinated_answer_fails_the_no_hallucination_case():
    case = AgentCase(
        id="outside",
        question="Japonya'nın başkenti neresidir?",
        expect_status="insufficient_context",
        answer_not_contains=["Tokyo"],
    )
    hallucinating = _answered_result(answer="Japonya'nın başkenti Tokyo'dur.")

    result = _run(evaluate_agent_case(FakeAgent(hallucinating), case))

    assert not result.passed
    failed = [a.type for a in result.assertion_results if not a.passed]
    assert "contains" in failed and "not_contains" in failed


# --- 3. sorgu iyilestirme ---


def test_refinement_case_detects_the_second_search():
    case = AgentCase(
        id="refine",
        question="bölge",
        expect_status="answered",
        expect_action="refined_query",
        expect_tool_calls=2,
    )

    result = _run(evaluate_agent_case(FakeAgent(_refined_result()), case))

    assert result.passed
    assert "actions: [assessed_question][searched_rag][refined_query]" in result.observation


def test_refinement_case_fails_when_no_second_search_happened():
    case = AgentCase(id="refine", question="bölge", expect_action="refined_query")

    result = _run(evaluate_agent_case(FakeAgent(_answered_result()), case))

    assert not result.passed


# --- 4. dogrudan cevap / tool kullanim davranisi ---


def test_direct_answer_case_requires_no_tool_use():
    case = AgentCase(
        id="direct",
        question="Merhaba",
        expect_status="answered_without_retrieval",
        expect_no_tool=True,
        expect_action="answered_without_retrieval",
    )

    result = _run(evaluate_agent_case(FakeAgent(_direct_result()), case))

    assert result.passed


def test_unnecessary_retrieval_fails_the_no_tool_case():
    case = AgentCase(id="direct", question="Merhaba", expect_no_tool=True)

    result = _run(evaluate_agent_case(FakeAgent(_answered_result()), case))

    assert not result.passed, "gereksiz RAG cagrisi yakalanmali"


def test_expected_tool_must_actually_have_been_called():
    case = AgentCase(id="tool", question="Soru?", expect_tool="rag_search")

    assert _run(evaluate_agent_case(FakeAgent(_answered_result()), case)).passed
    assert not _run(evaluate_agent_case(FakeAgent(_direct_result()), case)).passed


# --- 5. atif dogrulugu ---


def test_citation_regex_is_checked_against_the_answer():
    case = AgentCase(
        id="cite",
        question="Soru?",
        answer_regex=r"\[turkiye_cografyasi\.md::[0-9]+\]",
    )

    assert _run(evaluate_agent_case(FakeAgent(_answered_result()), case)).passed
    uncited = _answered_result(answer="Ağrı Dağı 5137 metredir.")
    assert not _run(evaluate_agent_case(FakeAgent(uncited), case)).passed


def test_expected_source_matches_the_sources_line_not_a_citation_in_the_answer():
    """Cevapta atif olarak gecen bir kimlik, gercekten kaynak listesinde
    olmadigi halde vakayi gecirmemeli."""
    case = AgentCase(id="src", question="Soru?", expect_source="uydurma.md::9")
    fabricated = AgentResult(
        answer="Cevap [uydurma.md::9].",
        status=AgentStatus.ANSWERED,
        sources=[_passage()],
        steps=[Step(AgentAction.ANSWERED_FROM_CONTEXT, "")],
        tool_calls=[_tool_call()],
    )

    result = _run(evaluate_agent_case(FakeAgent(fabricated), case))

    assert not result.passed


# --- LLM judge yalnizca gerektiginde ---


def test_no_judge_configured_means_no_judge_client_usage():
    case = AgentCase(id="nojudge", question="Soru?", expect_status="answered")
    judge = FakeJudgeClient()

    result = _run(evaluate_agent_case(FakeAgent(_answered_result()), case, judge_client=judge))

    assert result.passed
    assert judge.calls == 0, "judge yapilandirilmadiysa API cagrisi olmamali"


def test_judge_assertion_uses_the_injected_client():
    case = AgentCase(
        id="judge",
        question="Soru?",
        judge=JudgeSpec(rubric="Cevap kaynak gösteriyor mu?", threshold=3),
    )
    judge = FakeJudgeClient(score=4)

    result = _run(evaluate_agent_case(FakeAgent(_answered_result()), case, judge_client=judge))

    assert judge.calls == 1
    judge_results = [a for a in result.assertion_results if a.type == "llm_judge"]
    assert len(judge_results) == 1
    assert judge_results[0].passed
    assert judge_results[0].score == 4.0


def test_judge_below_threshold_fails_the_case():
    case = AgentCase(id="judge", question="Soru?", judge=JudgeSpec(rubric="r", threshold=4))

    result = _run(
        evaluate_agent_case(
            FakeAgent(_answered_result()), case, judge_client=FakeJudgeClient(score=2)
        )
    )

    assert not result.passed


def test_judge_error_is_reported_as_a_failed_assertion_not_a_crash():
    case = AgentCase(id="judge", question="Soru?", judge=JudgeSpec(rubric="r"))

    # judge_client=None -> promptevals net bir RuntimeError firlatir
    result = _run(evaluate_agent_case(FakeAgent(_answered_result()), case, judge_client=None))

    assert not result.passed
    assert isinstance(result.assertion_results[0], AssertionResult)
    assert "GEMINI_API_KEY" in result.assertion_results[0].detail


# --- kosucu / ozet ---


def test_agent_exception_is_captured_as_a_case_error():
    case = AgentCase(id="boom", question="Soru?", expect_status="answered")

    result = _run(evaluate_agent_case(FakeAgent(error=RuntimeError("patladi")), case))

    assert not result.passed
    assert "patladi" in result.error


def test_suite_summary_counts_and_latency():
    cases = [
        AgentCase(id="ok", question="q1", expect_status="answered"),
        AgentCase(id="bad", question="q2", expect_status="insufficient_context"),
    ]
    summary = _run(run_agent_eval(FakeAgent(_answered_result()), cases))

    assert summary.total == 2
    assert summary.passed == 1
    assert summary.failed == 1
    assert summary.avg_latency_ms >= 0.0


def test_llm_calls_are_tracked_per_case_when_a_counter_is_given():
    counter = {"n": 0}

    class _CountingAgent(FakeAgent):
        def run(self, question):
            counter["n"] += 3
            return super().run(question)

    case = AgentCase(id="c", question="q", expect_status="answered")
    result = _run(
        evaluate_agent_case(
            _CountingAgent(_answered_result()), case, call_count=lambda: counter["n"]
        )
    )

    assert result.llm_calls == 3


# --- gonderilen ornek suite ---


def test_shipped_suite_loads_and_every_case_compiles():
    suite = load_suite(SUITE_PATH)

    assert len(suite.cases) == 6
    ids = [case.id for case in suite.cases]
    assert len(set(ids)) == len(ids)
    for case in suite.cases:
        assert compile_assertions(case), f"{case.id} hic assertion uretmedi"


def test_shipped_suite_spends_at_most_one_judge_call():
    suite = load_suite(SUITE_PATH)

    judge_assertions = [
        compiled
        for case in suite.cases
        for compiled in compile_assertions(case)
        if compiled.assertion.type == "llm_judge"
    ]

    assert len(judge_assertions) == 1, "maliyet kontrolu: yalnizca bir judge cagrisi beklenir"


def test_shipped_suite_covers_the_required_behaviours():
    cases = {case.id: case for case in load_suite(SUITE_PATH).cases}

    assert any(c.expect_status == "answered" and c.expect_tool for c in cases.values())
    assert any(c.expect_status == "insufficient_context" for c in cases.values())
    assert any(c.expect_no_tool for c in cases.values())
    assert any(c.answer_regex for c in cases.values())
    assert any(c.expect_source for c in cases.values())
    assert any(c.judge for c in cases.values())
