"""GeminiAgentLLM adapter testleri.

Gercek Gemini API'ye hicbir cagri yapilmaz: client fake'lenir. API key
gerekmez.
"""

import pytest

from rag_tr.agent.config import AgentSettings
from rag_tr.agent.contracts import AgentStatus, SearchOutcome, ToolStatus
from rag_tr.agent.gemini import (
    GeminiAgentLLM,
    PassageAssessmentOut,
    RetrievalDecisionOut,
    SynthesisOut,
)
from rag_tr.agent.loop import ResearchAgent
from rag_tr.contracts import Passage


def _passage(chunk_id="cografya.md::0", text="Türkiye'nin başkenti Ankara'dır.", page=None, rank=1):
    return Passage(
        chunk_id=chunk_id,
        source_file=chunk_id.split("::")[0],
        page_number=page,
        text=text,
        rank=rank,
        score=1.0 / rank,
    )


class _FakeResponse:
    def __init__(self, parsed=None, text="") -> None:
        self.parsed = parsed
        self.text = text


class _FakeModels:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if not self._responses:
            raise AssertionError("beklenenden fazla Gemini cagrisi yapildi")
        return self._responses.pop(0)


class FakeGeminiClient:
    def __init__(self, responses) -> None:
        self.models = _FakeModels(responses)


def _llm(responses, **kwargs) -> GeminiAgentLLM:
    return GeminiAgentLLM(FakeGeminiClient(responses), model="gemini-2.5-flash", **kwargs)


# --- karar 1: arama gerekli mi ---


def test_needs_retrieval_maps_structured_output():
    llm = _llm([_FakeResponse(RetrievalDecisionOut(needs_retrieval=True, search_query="Türkiye başkent", reason="doküman bilgisi"))])

    decision = llm.needs_retrieval("Türkiye'nin başkenti neresidir?")

    assert decision.needs_retrieval is True
    assert decision.search_query == "Türkiye başkent"
    assert decision.reason == "doküman bilgisi"


def test_needs_retrieval_handles_no_search_query():
    llm = _llm([_FakeResponse(RetrievalDecisionOut(needs_retrieval=False, reason="selamlama"))])

    decision = llm.needs_retrieval("Merhaba")

    assert decision.needs_retrieval is False
    assert decision.search_query is None


def test_needs_retrieval_falls_back_to_searching_when_parsing_fails():
    """Ayristirma basarisizsa guvenli taraf: aramayi dene, uydurma yapma."""
    llm = _llm([_FakeResponse(parsed=None, text="bozuk cikti")])

    decision = llm.needs_retrieval("Başkent neresi?")

    assert decision.needs_retrieval is True
    assert decision.search_query is None


# --- karar 2: pasajlar yeterli mi ---


def test_assess_passages_maps_structured_output():
    llm = _llm([_FakeResponse(PassageAssessmentOut(sufficient=True, reason="yeterli"))])

    assessment = llm.assess_passages("Başkent?", [_passage()])

    assert assessment.sufficient is True
    assert assessment.refined_query is None


def test_assess_passages_can_request_a_refined_search():
    llm = _llm(
        [_FakeResponse(PassageAssessmentOut(sufficient=False, refined_query="Ankara başkent", reason="çok genel"))]
    )

    assessment = llm.assess_passages("başkent", [_passage()])

    assert assessment.sufficient is False
    assert assessment.refined_query == "Ankara başkent"


def test_assess_passages_falls_back_to_insufficient_when_parsing_fails():
    llm = _llm([_FakeResponse(parsed=None, text="bozuk")])

    assessment = llm.assess_passages("Soru?", [_passage()])

    assert assessment.sufficient is False, "ayristirilamayan yanit cevap uretmeye izin vermemeli"


def test_assess_passages_with_no_passages_does_not_call_gemini():
    llm = _llm([])

    assessment = llm.assess_passages("Soru?", [])

    assert assessment.sufficient is False
    assert llm.call_count == 0, "bos pasaj listesi icin API cagrisi gereksiz"


# --- sentez ---


def test_synthesize_returns_the_structured_answer():
    llm = _llm([_FakeResponse(SynthesisOut(answer="Ankara'dır [cografya.md::0]."))])

    answer = llm.synthesize("Başkent?", [_passage()])

    assert answer == "Ankara'dır [cografya.md::0]."


def test_synthesize_prompt_lists_the_allowed_chunk_ids_and_grounding_rule():
    client = FakeGeminiClient([_FakeResponse(SynthesisOut(answer="x"))])
    llm = GeminiAgentLLM(client, model="gemini-2.5-flash")

    llm.synthesize("Başkent?", [_passage("cografya.md::0"), _passage("tarih.md::3", page=7, rank=2)])

    call = client.models.calls[0]
    assert "cografya.md::0" in call["contents"]
    assert "tarih.md::3" in call["contents"]
    assert "page 7" in call["contents"]
    system = call["config"].system_instruction
    # Prompt artik dil-notr (Ingilizce) ve sonuna dil direktifi ekleniyor;
    # garantiler ayni: yalnizca verilen pasajlar ve kaynak uydurma yasagi.
    assert "only on the passages" in system.lower()
    assert "never invent a source" in system.lower()
    assert system.rstrip().endswith("Kullanıcıya Türkçe cevap ver.")


def test_synthesize_falls_back_to_plain_text_when_parsing_fails():
    llm = _llm([_FakeResponse(parsed=None, text="Düz metin cevap")])

    assert llm.synthesize("Soru?", [_passage()]) == "Düz metin cevap"


# --- dogrudan cevap ---


def test_answer_directly_returns_plain_text():
    llm = _llm([_FakeResponse(text="Merhaba, nasıl yardımcı olabilirim?")])

    assert llm.answer_directly("Merhaba") == "Merhaba, nasıl yardımcı olabilirim?"


# --- determinizm / yapilandirma ---


def test_requests_are_deterministic_and_use_the_configured_model():
    client = FakeGeminiClient([_FakeResponse(RetrievalDecisionOut(needs_retrieval=False))])
    llm = GeminiAgentLLM(client, model="gemini-2.5-flash")

    llm.needs_retrieval("Soru?")

    call = client.models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["config"].temperature == 0.0
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema is RetrievalDecisionOut


def test_call_count_tracks_every_gemini_request():
    llm = _llm(
        [
            _FakeResponse(RetrievalDecisionOut(needs_retrieval=True)),
            _FakeResponse(PassageAssessmentOut(sufficient=True)),
            _FakeResponse(SynthesisOut(answer="cevap")),
        ]
    )

    llm.needs_retrieval("q")
    llm.assess_passages("q", [_passage()])
    llm.synthesize("q", [_passage()])

    assert llm.call_count == 3


def test_adapter_satisfies_the_agent_llm_surface():
    llm = _llm([])
    for name in ("needs_retrieval", "assess_passages", "synthesize", "answer_directly"):
        assert callable(getattr(llm, name))


# --- ayarlar ---


def test_default_model_is_a_free_tier_gemini_model():
    assert AgentSettings(gemini_api_key=None).gemini_agent_model == "gemini-2.5-flash"


def test_from_env_raises_a_clear_error_without_a_key():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiAgentLLM.from_env(AgentSettings(gemini_api_key=None))


def test_from_env_builds_a_client_with_the_configured_model(monkeypatch):
    created: dict = {}

    class _StubClient:
        def __init__(self, api_key):
            created["api_key_len"] = len(api_key)

    monkeypatch.setattr("google.genai.Client", _StubClient)

    llm = GeminiAgentLLM.from_env(
        AgentSettings(gemini_api_key="fake-key-value", gemini_agent_model="gemini-2.5-flash-lite")
    )

    assert llm.model == "gemini-2.5-flash-lite"
    assert created["api_key_len"] == len("fake-key-value")


# --- agent dongusuyle entegrasyon (agent Gemini'yi bilmiyor) ---


class _ScriptedSearchTool:
    name = "rag_search"

    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.queries: list[str] = []

    def search(self, query, top_k=5, source_file=None):
        self.queries.append(query)
        return self.outcomes.pop(0) if self.outcomes else SearchOutcome(status=ToolStatus.EMPTY)


def test_research_agent_runs_end_to_end_on_the_gemini_adapter():
    llm = _llm(
        [
            _FakeResponse(RetrievalDecisionOut(needs_retrieval=True, search_query="Türkiye başkent")),
            _FakeResponse(PassageAssessmentOut(sufficient=True)),
            _FakeResponse(SynthesisOut(answer="Ankara'dır [cografya.md::0].")),
        ]
    )
    tool = _ScriptedSearchTool([SearchOutcome(status=ToolStatus.OK, passages=[_passage()])])

    result = ResearchAgent(llm=llm, search_tool=tool).run("Türkiye'nin başkenti neresidir?")

    assert result.status is AgentStatus.ANSWERED
    assert result.answer == "Ankara'dır [cografya.md::0]."
    assert [s.chunk_id for s in result.sources] == ["cografya.md::0"]
    assert tool.queries == ["Türkiye başkent"]
    assert llm.call_count == 3


def test_research_agent_skips_retrieval_with_a_single_gemini_call_for_the_decision():
    llm = _llm(
        [
            _FakeResponse(RetrievalDecisionOut(needs_retrieval=False, reason="selamlama")),
            _FakeResponse(text="Merhaba!"),
        ]
    )
    tool = _ScriptedSearchTool([])

    result = ResearchAgent(llm=llm, search_tool=tool).run("Merhaba")

    assert result.status is AgentStatus.ANSWERED_WITHOUT_RETRIEVAL
    assert result.answer == "Merhaba!"
    assert tool.queries == []
    assert llm.call_count == 2


class _RateLimitError(Exception):
    """google.genai.errors.APIError yerine gecen, kod tasiyan sahte hata."""

    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


class _FlakyModels(_FakeModels):
    def __init__(self, failures: list[Exception], responses) -> None:
        super().__init__(responses)
        self._failures = list(failures)

    def generate_content(self, *, model, contents, config):
        if self._failures:
            raise self._failures.pop(0)
        return super().generate_content(model=model, contents=contents, config=config)


class _FlakyClient:
    def __init__(self, failures, responses) -> None:
        self.models = _FlakyModels(failures, responses)


def _flaky_llm(failures, responses, **kwargs):
    slept: list[float] = []
    llm = GeminiAgentLLM(
        _FlakyClient(failures, responses),
        model="gemini-2.5-flash",
        sleep=slept.append,
        retry_delay_seconds=0.01,
        **kwargs,
    )
    return llm, slept


def test_rate_limit_is_retried_without_failing_the_run(monkeypatch):
    monkeypatch.setattr("rag_tr.agent.gemini.genai_errors.APIError", _RateLimitError)
    llm, slept = _flaky_llm(
        [_RateLimitError(429)], [_FakeResponse(RetrievalDecisionOut(needs_retrieval=True))]
    )

    decision = llm.needs_retrieval("Soru?")

    assert decision.needs_retrieval is True
    assert llm.retry_count == 1
    assert len(slept) == 1


def test_server_error_is_retried(monkeypatch):
    monkeypatch.setattr("rag_tr.agent.gemini.genai_errors.APIError", _RateLimitError)
    llm, _ = _flaky_llm(
        [_RateLimitError(503)], [_FakeResponse(RetrievalDecisionOut(needs_retrieval=False))]
    )

    llm.needs_retrieval("Soru?")

    assert llm.retry_count == 1


def test_client_error_is_not_retried(monkeypatch):
    monkeypatch.setattr("rag_tr.agent.gemini.genai_errors.APIError", _RateLimitError)
    llm, _ = _flaky_llm([_RateLimitError(400)], [])

    with pytest.raises(_RateLimitError):
        llm.needs_retrieval("Soru?")

    assert llm.retry_count == 0


def test_retries_are_bounded(monkeypatch):
    monkeypatch.setattr("rag_tr.agent.gemini.genai_errors.APIError", _RateLimitError)
    llm, slept = _flaky_llm([_RateLimitError(429)] * 10, [], max_retries=2)

    with pytest.raises(_RateLimitError):
        llm.needs_retrieval("Soru?")

    assert llm.retry_count == 2
    assert len(slept) == 2
