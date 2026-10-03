"""Cevap dili destegi.

Yedi arayuz dili var ve secilen dil `/agent/ask` ile birlikte geliyor. Bu
testler uc seyi guvenceye alir:

1. Dil paketleri eksiksiz ve sablonlari tutarli.
2. Kullaniciya gorunen sabit metinler gercekten secilen dilden geliyor.
3. Dil destegi EK bir model cagrisi uretmiyor -- sorunun once cevrilip sonra
   sorulmasi gibi bir ikinci tur yok; dil yalnizca mevcut cagrilarin system
   prompt'una giren bir cumle.

Hicbir canli API cagrisi yapilmaz.
"""

import pytest
from pydantic import ValidationError

from rag_tr.agent.contracts import (
    AgentAction,
    AgentStatus,
    PassageAssessment,
    RetrievalDecision,
    ToolStatus,
)
from rag_tr.agent.gemini import (
    _ASSESS_SYSTEM,
    _DECIDE_SYSTEM,
    _DIRECT_SYSTEM,
    _SYNTHESIZE_SYSTEM,
    GeminiAgentLLM,
    _with_language,
)
from rag_tr.agent.loop import ResearchAgent
from rag_tr.agent.tools import SearchOutcome
from rag_tr.api.agent_schemas import AgentAskRequest
from rag_tr.contracts import Passage
from rag_tr.i18n import (
    CORPUS_LANGUAGE_NAME,
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    get_language_pack,
    normalize_language,
)

EXPECTED_LANGUAGES = ("tr", "en", "fr", "ar", "es", "zh", "hi")

TEXT_FIELDS = (
    "name",
    "directive",
    "no_context",
    "tool_failure",
    "search_needed",
    "search_not_needed",
    "answered_without_retrieval",
    "query_refined",
    "decision_unparsed",
    "assessment_unparsed",
    "no_passages_found",
)


# --- dil paketleri ----------------------------------------------------------


def test_exactly_the_seven_interface_languages_are_supported():
    assert SUPPORTED_LANGUAGES == EXPECTED_LANGUAGES


def test_the_default_language_is_the_corpus_language():
    assert DEFAULT_LANGUAGE == "tr"


@pytest.mark.parametrize("code", EXPECTED_LANGUAGES)
def test_every_language_pack_is_complete(code):
    pack = get_language_pack(code)

    assert pack.code == code
    for field in TEXT_FIELDS:
        value = getattr(pack, field)
        assert isinstance(value, str) and value.strip(), f"{code}.{field} bos"


@pytest.mark.parametrize("code", EXPECTED_LANGUAGES)
def test_templates_carry_exactly_the_placeholders_the_loop_fills(code):
    """Eksik bir alan adi KeyError, fazlasi ise cevapta sizan suslu parantez
    olurdu; ikisi de yalnizca o dil secildiginde ortaya cikardi."""
    pack = get_language_pack(code)

    assert pack.searched_attempt.format(attempt=1, count=2)
    assert "{" not in pack.searched_attempt.format(attempt=1, count=2)
    assert pack.answered_from_context.format(count=3)
    assert "{" not in pack.answered_from_context.format(count=3)
    assert pack.insufficient_after_searches.format(searches=2)
    assert "{" not in pack.insufficient_after_searches.format(searches=2)
    assert pack.tool_error.format(error="x")
    assert "{" not in pack.tool_error.format(error="x")


def test_each_language_has_its_own_wording():
    """Kopyala-yapistir ile ayni metnin iki dile girmesini yakalar."""
    directives = {get_language_pack(code).directive for code in EXPECTED_LANGUAGES}
    no_context = {get_language_pack(code).no_context for code in EXPECTED_LANGUAGES}

    assert len(directives) == len(EXPECTED_LANGUAGES)
    assert len(no_context) == len(EXPECTED_LANGUAGES)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("tr", "tr"),
        ("EN", "en"),
        ("en-US", "en"),
        ("zh_Hans", "zh"),
        ("  fr  ", "fr"),
        ("de", "tr"),
        ("", "tr"),
        (None, "tr"),
    ],
)
def test_language_codes_are_normalized(given, expected):
    assert normalize_language(given) == expected


# --- istek semasi ----------------------------------------------------------


def test_language_defaults_to_the_corpus_language():
    assert AgentAskRequest(question="soru").language == DEFAULT_LANGUAGE


@pytest.mark.parametrize("code", EXPECTED_LANGUAGES)
def test_every_supported_language_is_accepted(code):
    assert AgentAskRequest(question="soru", language=code).language == code


def test_a_regional_code_is_reduced_to_its_language():
    assert AgentAskRequest(question="soru", language="en-GB").language == "en"


def test_an_unsupported_language_is_rejected_instead_of_falling_back():
    with pytest.raises(ValidationError) as excinfo:
        AgentAskRequest(question="soru", language="de")

    message = str(excinfo.value)
    assert "de" in message
    # Hata, hangi dillerin desteklendigini de soylemeli.
    assert "fr" in message and "hi" in message


# --- system prompt'lar -----------------------------------------------------


@pytest.mark.parametrize("code", EXPECTED_LANGUAGES)
@pytest.mark.parametrize(
    "system", [_DECIDE_SYSTEM, _ASSESS_SYSTEM, _SYNTHESIZE_SYSTEM, _DIRECT_SYSTEM]
)
def test_the_language_directive_is_appended_to_every_system_prompt(code, system):
    pack = get_language_pack(code)

    combined = _with_language(system, pack)

    assert combined.startswith(system)
    assert combined.endswith(pack.directive)


def test_the_search_query_is_requested_in_the_corpus_language_not_the_user_language():
    """Retrieval'in calismaya devam etmesinin sarti.

    BM25 tam token esleymesi yapiyor: Fransizca yazilmis bir sorgu Turkce bir
    dokumani anahtar kelimeyle bulamaz. Bu yuzden sorgu korpus dilinde
    istenir, cevap ise kullanicinin dilinde verilir."""
    for system in (_DECIDE_SYSTEM, _ASSESS_SYSTEM):
        assert CORPUS_LANGUAGE_NAME in system

    french = _with_language(_DECIDE_SYSTEM, get_language_pack("fr"))
    assert CORPUS_LANGUAGE_NAME in french
    assert "français" in french


def test_citations_are_explicitly_protected_from_translation():
    assert "do not translate" in _SYNTHESIZE_SYSTEM.lower()


# --- LLM sarmalayicisi -----------------------------------------------------


class _FakeClient:
    """Yalnizca kimlik karsilastirmasi icin; hicbir sey cagrilmaz."""


def test_for_language_shares_the_client_instead_of_building_a_new_one():
    client = _FakeClient()
    llm = GeminiAgentLLM(client, "fake-model")

    french = llm.for_language("fr")

    assert french is not llm
    assert french.language == "fr"
    assert french._client is client, "istek basina yeni Gemini client kurulmamali"
    assert french.model == llm.model


def test_for_language_returns_the_same_instance_for_the_same_language():
    llm = GeminiAgentLLM(_FakeClient(), "fake-model", language="en")

    assert llm.for_language("en") is llm
    assert llm.for_language("en-US") is llm


# --- dongu: kullaniciya gorunen metinler ----------------------------------


class _StubLLM:
    """Sabit kararlar dondurur ve cagri sayar. Gercek LLM cagrisi yok."""

    def __init__(self, *, needs=True, sufficient=True, reason=""):
        self.needs = needs
        self.sufficient = sufficient
        self.reason = reason
        self.calls = 0

    def needs_retrieval(self, question):
        self.calls += 1
        return RetrievalDecision(
            needs_retrieval=self.needs, search_query="osmanlı kuruluş", reason=self.reason
        )

    def assess_passages(self, question, passages):
        self.calls += 1
        return PassageAssessment(
            sufficient=self.sufficient, refined_query=None, reason=self.reason
        )

    def synthesize(self, question, passages):
        self.calls += 1
        return "cevap [a.md::0]"

    def answer_directly(self, question):
        self.calls += 1
        return "direkt cevap"


class _StubTool:
    name = "rag_search"

    def __init__(self, outcome):
        self.outcome = outcome

    def search(self, query, top_k):
        return self.outcome


def _passage() -> Passage:
    return Passage(
        chunk_id="a.md::0",
        source_file="a.md",
        page_number=None,
        text="Osmanlı Devleti 1299 yılında kuruldu.",
        rank=1,
        score=0.5,
    )


def _ok_tool() -> _StubTool:
    return _StubTool(SearchOutcome(status=ToolStatus.OK, passages=[_passage()], error=None))


@pytest.mark.parametrize("code", EXPECTED_LANGUAGES)
def test_step_details_are_written_in_the_requested_language(code):
    pack = get_language_pack(code)
    agent = ResearchAgent(llm=_StubLLM(), search_tool=_ok_tool(), language=code)

    result = agent.run("Osmanlı Devleti hangi yılda kuruldu?")

    assert agent.language == code
    details = {step.action: step.detail for step in result.steps}
    assert details[AgentAction.SEARCHED_RAG] == pack.searched_attempt.format(
        attempt=1, count=1
    )
    assert details[AgentAction.ANSWERED_FROM_CONTEXT] == pack.answered_from_context.format(
        count=1
    )


@pytest.mark.parametrize("code", EXPECTED_LANGUAGES)
def test_the_no_context_answer_is_written_in_the_requested_language(code):
    empty = _StubTool(SearchOutcome(status=ToolStatus.OK, passages=[], error=None))
    agent = ResearchAgent(
        llm=_StubLLM(sufficient=False), search_tool=empty, language=code
    )

    result = agent.run("korpus dışı bir soru")

    assert result.status is AgentStatus.INSUFFICIENT_CONTEXT
    assert result.answer == get_language_pack(code).no_context


@pytest.mark.parametrize("code", EXPECTED_LANGUAGES)
def test_the_tool_failure_answer_is_written_in_the_requested_language(code):
    failed = _StubTool(SearchOutcome(status=ToolStatus.FAILED, passages=[], error="boom"))
    agent = ResearchAgent(llm=_StubLLM(), search_tool=failed, language=code)

    result = agent.run("soru")

    assert result.status is AgentStatus.TOOL_FAILURE
    assert result.answer == get_language_pack(code).tool_failure


@pytest.mark.parametrize("code", EXPECTED_LANGUAGES)
def test_a_model_supplied_reason_is_preferred_over_the_fallback(code):
    """Gerekce modelden gelir ve dil direktifi sayesinde zaten dogru dildedir;
    yedek metin yalnizca model bos dondurdugunde kullanilir."""
    agent = ResearchAgent(
        llm=_StubLLM(reason="model gerekcesi"), search_tool=_ok_tool(), language=code
    )

    result = agent.run("soru")

    assessed = next(
        step for step in result.steps if step.action is AgentAction.ASSESSED_QUESTION
    )
    assert assessed.detail == "model gerekcesi"


def test_changing_the_language_does_not_add_a_single_model_call():
    """Dil destegi, dil basina ek bir generation cagrisi URETMEMELI.

    Sorunun once baska bir dile cevrilip sonra tekrar cevrilmesi gibi bir
    ikinci tur yok: dil yalnizca mevcut cagrilarin system prompt'una giren
    bir cumle."""
    counts = {}
    for code in EXPECTED_LANGUAGES:
        llm = _StubLLM()
        ResearchAgent(llm=llm, search_tool=_ok_tool(), language=code).run("soru")
        counts[code] = llm.calls

    assert len(set(counts.values())) == 1, counts
    # ASSESS + INSPECT + SYNTHESIZE = 3; dilden bagimsiz.
    assert set(counts.values()) == {3}


def test_retrieval_query_is_not_rewritten_by_the_language_choice():
    """Hibrit retrieval korunur: tool'a giden sorgu, dilden bagimsiz olarak
    modelin urettigi korpus-dili sorgudur."""
    seen = []

    class _RecordingTool(_StubTool):
        def search(self, query, top_k):
            seen.append(query)
            return self.outcome

    for code in EXPECTED_LANGUAGES:
        tool = _RecordingTool(
            SearchOutcome(status=ToolStatus.OK, passages=[_passage()], error=None)
        )
        ResearchAgent(llm=_StubLLM(), search_tool=tool, language=code).run("soru")

    assert set(seen) == {"osmanlı kuruluş"}
