"""POST /agent/ask testleri.

Gemini, Anthropic, ag ve gercek embedding modeli kullanilmaz: hem RAG servisi
hem de ResearchAgent fake olarak enjekte edilir.
"""

import threading

import pytest
from fastapi.testclient import TestClient

from rag_tr.agent.contracts import (
    AgentAction,
    AgentResult,
    AgentStatus,
    Step,
    ToolCall,
    ToolStatus,
)
from rag_tr.api import agent_routes, deps
from rag_tr.agent.loop import DEFAULT_TOP_K
from rag_tr.api.main import create_app
from rag_tr.contracts import NO_CONTEXT_MESSAGE, ErrorCode, Passage

MAIN_THREAD_ID = threading.get_ident()


def _passage(chunk_id="turkiye_cografyasi.md::0", page=None, rank=1):
    return Passage(
        chunk_id=chunk_id,
        source_file=chunk_id.split("::")[0],
        page_number=page,
        text="Ağrı Dağı 5137 metredir.",
        rank=rank,
        score=0.0328,
    )


def _answered():
    return AgentResult(
        answer="Ağrı Dağı, 5137 metre [turkiye_cografyasi.md::0].",
        status=AgentStatus.ANSWERED,
        sources=[_passage(), _passage("osmanli_tarihi.md::0", page=3, rank=2)],
        steps=[
            Step(AgentAction.ASSESSED_QUESTION, "olgusal bilgi gerekiyor"),
            Step(AgentAction.SEARCHED_RAG, "deneme 1: 2 pasaj bulundu"),
            Step(AgentAction.ANSWERED_FROM_CONTEXT, "2 pasaj temel alindi"),
        ],
        tool_calls=[
            ToolCall(
                tool="rag_search",
                query="Türkiye en yüksek dağ",
                top_k=5,
                source_file=None,
                status=ToolStatus.OK,
                passages_found=2,
            )
        ],
    )


class FakeAgent:
    """ResearchAgent.run(question) sozlesmesinin tamami."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self._result = result if result is not None else _answered()
        self._error = error
        self.questions: list[str] = []
        self.thread_ids: list[int] = []

    def run(self, question: str) -> AgentResult:
        self.questions.append(question)
        self.thread_ids.append(threading.get_ident())
        if self._error is not None:
            raise self._error
        return self._result


class _StubVectorStore:
    """Agent uc noktasi, bos korpusta agent'i hic calistirmamak icin chunk
    sayisina bakiyor."""

    def __init__(self, count: int = 4) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class _StubService:
    """Agent uc noktasinin dokundugu yuzey: yalnizca vector_store.count()."""

    def __init__(self, chunk_count: int = 4) -> None:
        self.vector_store = _StubVectorStore(chunk_count)


@pytest.fixture
def agent() -> FakeAgent:
    return FakeAgent()


@pytest.fixture
def client(agent) -> TestClient:
    return TestClient(create_app(service=_StubService(), agent=agent))


# --- temel akis ---


def test_ask_returns_the_agent_result(client, agent):
    response = client.post("/agent/ask", json={"question": "Türkiye'nin en yüksek dağı?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Ağrı Dağı, 5137 metre [turkiye_cografyasi.md::0]."
    assert body["status"] == "answered"
    assert agent.questions == ["Türkiye'nin en yüksek dağı?"]


def test_dependency_injection_uses_the_injected_agent(agent):
    """Enjekte edilen agent gercekten cagrilmali, yenisi kurulmamali."""
    local = TestClient(create_app(service=_StubService(), agent=agent))

    local.post("/agent/ask", json={"question": "soru bir"})
    local.post("/agent/ask", json={"question": "soru iki"})

    assert agent.questions == ["soru bir", "soru iki"]


def test_blocking_run_is_dispatched_through_the_threadpool(client, agent, monkeypatch):
    calls: list = []
    original = agent_routes.run_in_threadpool

    async def _spy(func, *args, **kwargs):
        calls.append(func)
        return await original(func, *args, **kwargs)

    monkeypatch.setattr(agent_routes, "run_in_threadpool", _spy)

    assert client.post("/agent/ask", json={"question": "soru"}).status_code == 200

    assert calls == [agent.run], "agent.run threadpool uzerinden cagrilmali"
    assert agent.thread_ids[0] != MAIN_THREAD_ID, "bloklayici cagri event loop'u tutmamali"


# --- sema eslemesi ---


def test_sources_are_mapped_field_by_field(client):
    body = client.post("/agent/ask", json={"question": "soru"}).json()

    assert body["sources"] == [
        {
            "chunk_id": "turkiye_cografyasi.md::0",
            "source_file": "turkiye_cografyasi.md",
            "page_number": None,
            "text": "Ağrı Dağı 5137 metredir.",
            "rank": 1,
            "score": 0.0328,
        },
        {
            "chunk_id": "osmanli_tarihi.md::0",
            "source_file": "osmanli_tarihi.md",
            "page_number": 3,
            "text": "Ağrı Dağı 5137 metredir.",
            "rank": 2,
            "score": 0.0328,
        },
    ]


def test_steps_are_mapped_as_safe_action_labels(client):
    body = client.post("/agent/ask", json={"question": "soru"}).json()

    assert body["steps"] == [
        {"action": "assessed_question", "detail": "olgusal bilgi gerekiyor"},
        {"action": "searched_rag", "detail": "deneme 1: 2 pasaj bulundu"},
        {"action": "answered_from_context", "detail": "2 pasaj temel alindi"},
    ]


def test_tool_calls_are_mapped_field_by_field(client):
    body = client.post("/agent/ask", json={"question": "soru"}).json()

    assert body["tool_calls"] == [
        {
            "tool": "rag_search",
            "query": "Türkiye en yüksek dağ",
            "top_k": 5,
            "source_file": None,
            "status": "ok",
            "passages_found": 2,
        }
    ]


def test_response_exposes_only_the_contracted_fields(client):
    body = client.post("/agent/ask", json={"question": "soru"}).json()

    assert set(body) == {"answer", "status", "sources", "steps", "tool_calls"}


# --- agent'in diger sonuc durumlari ---


def test_insufficient_context_is_returned_with_its_status():
    result = AgentResult(
        answer=NO_CONTEXT_MESSAGE,
        status=AgentStatus.INSUFFICIENT_CONTEXT,
        sources=[],
        steps=[Step(AgentAction.DECLARED_INSUFFICIENT_CONTEXT, "yeterli baglam yok")],
        tool_calls=[],
    )
    local = TestClient(create_app(service=_StubService(), agent=FakeAgent(result)))

    body = local.post("/agent/ask", json={"question": "Japonya?"}).json()

    assert body["status"] == "insufficient_context"
    assert body["answer"] == NO_CONTEXT_MESSAGE
    assert body["sources"] == []


def test_answered_without_retrieval_is_returned_with_its_status():
    result = AgentResult(
        answer="Merhaba!",
        status=AgentStatus.ANSWERED_WITHOUT_RETRIEVAL,
        sources=[],
        steps=[Step(AgentAction.ANSWERED_WITHOUT_RETRIEVAL, "")],
        tool_calls=[],
    )
    local = TestClient(create_app(service=_StubService(), agent=FakeAgent(result)))

    body = local.post("/agent/ask", json={"question": "Merhaba"}).json()

    assert body["status"] == "answered_without_retrieval"
    assert body["tool_calls"] == []


def test_tool_failure_is_a_200_result_not_an_http_error():
    """TOOL_FAILURE agent'in kendi sonucu; istemci trace'i gorebilmeli."""
    result = AgentResult(
        answer="Belge aramasi sirasinda bir hata olustu.",
        status=AgentStatus.TOOL_FAILURE,
        sources=[],
        steps=[Step(AgentAction.TOOL_FAILED, "retrieval tool hatasi: chroma down")],
        tool_calls=[
            ToolCall(
                tool="rag_search",
                query="soru",
                top_k=5,
                source_file=None,
                status=ToolStatus.FAILED,
                passages_found=0,
            )
        ],
    )
    local = TestClient(create_app(service=_StubService(), agent=FakeAgent(result)))

    response = local.post("/agent/ask", json={"question": "soru"})

    assert response.status_code == 200
    assert response.json()["status"] == "tool_failure"
    assert response.json()["tool_calls"][0]["status"] == "failed"


# --- istek dogrulama ---


def test_empty_question_is_rejected(client, agent):
    response = client.post("/agent/ask", json={"question": ""})

    assert response.status_code == 422
    assert agent.questions == []


def test_missing_question_is_rejected(client, agent):
    assert client.post("/agent/ask", json={}).status_code == 422
    assert agent.questions == []


@pytest.mark.parametrize("bad_top_k", [0, -1])
def test_non_positive_top_k_is_rejected(client, agent, bad_top_k):
    response = client.post("/agent/ask", json={"question": "soru", "top_k": bad_top_k})

    assert response.status_code == 422
    assert agent.questions == []


def test_valid_top_k_is_accepted(client):
    assert client.post("/agent/ask", json={"question": "soru", "top_k": 3}).status_code == 200


def test_custom_top_k_uses_the_factory_without_rebuilding_the_service(agent):
    """Istege ozel top_k, ayni LLM/tool'u paylasan hafif bir agent uretmeli."""
    built: list[tuple[int, str]] = []
    custom = FakeAgent()

    def factory(top_k: int, language: str = "tr"):
        built.append((top_k, language))
        return custom

    app = create_app(service=_StubService(), agent=agent)
    app.state.agent_factory = factory
    local = TestClient(app)

    local.post("/agent/ask", json={"question": "soru", "top_k": 2})

    assert built == [(2, "tr")]
    assert custom.questions == ["soru"]
    assert agent.questions == [], "varsayilan agent bu istekte kullanilmamali"


def test_a_non_default_language_is_passed_through_to_the_factory(agent):
    """Secilen dil, istek basina kurulan agent'a gecmeli."""
    built: list[tuple[int, str]] = []
    custom = FakeAgent()

    def factory(top_k: int, language: str = "tr"):
        built.append((top_k, language))
        return custom

    app = create_app(service=_StubService(), agent=agent)
    app.state.agent_factory = factory
    local = TestClient(app)

    local.post("/agent/ask", json={"question": "soru", "language": "fr"})

    assert built == [(DEFAULT_TOP_K, "fr")], "dil fabrikaya varsayilan top_k ile gecmeli"
    assert custom.questions == ["soru"]


def test_an_unsupported_language_is_rejected_before_the_agent_runs(agent):
    app = create_app(service=_StubService(), agent=agent)
    local = TestClient(app)

    response = local.post("/agent/ask", json={"question": "soru", "language": "de"})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"
    assert agent.questions == [], "gecersiz dilde agent hic calismamali"


def test_default_top_k_does_not_use_the_factory(agent):
    built: list[int] = []
    app = create_app(service=_StubService(), agent=agent)
    app.state.agent_factory = lambda top_k, language="tr": built.append(top_k)
    local = TestClient(app)

    local.post("/agent/ask", json={"question": "soru"})

    assert built == []
    assert agent.questions == ["soru"]


# --- hata eslemesi ---


class _FakeGeminiError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


def _client_with_error(exc: Exception, monkeypatch) -> TestClient:
    monkeypatch.setattr(agent_routes.genai_errors, "APIError", _FakeGeminiError)
    return TestClient(
        create_app(service=_StubService(), agent=FakeAgent(error=exc)),
        raise_server_exceptions=False,
    )


def test_quota_exhaustion_maps_to_429_with_its_own_code(monkeypatch):
    local = _client_with_error(_FakeGeminiError(429), monkeypatch)

    response = local.post("/agent/ask", json={"question": "soru"})

    assert response.status_code == 429
    assert response.json()["detail"]["code"] == ErrorCode.QUOTA_EXHAUSTED.value


def test_other_gemini_api_errors_map_to_502_not_quota(monkeypatch):
    local = _client_with_error(_FakeGeminiError(500), monkeypatch)

    response = local.post("/agent/ask", json={"question": "soru"})

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == ErrorCode.GENERATION_ERROR.value


# Gercek bir uretim gozleminden geliyor: kota dolduguna dair 429 yanitinin
# `message` alani Gemini'nin ham hata govdesini oldugu gibi tasiyordu -- kota
# metrik adlari, proje katmani, yeniden deneme suresi ve dokumantasyon
# baglantilari dahil yaklasik 1200 karakter. Bu hem kullaniciya gosterilemeyecek
# bir metin, hem de istemciye sizmasi gerekmeyen altyapi ayrintisi. Ayrinti
# sunucu logunda kalmali; istemci kisa ve sabit bir cumle gormeli.

_UPSTREAM_NOISE = (
    "RESOURCE_EXHAUSTED {'error': {'code': 429, 'message': 'You exceeded your current "
    "quota', 'details': [{'quotaMetric': 'generativelanguage.googleapis.com/"
    "generate_content_free_tier_requests', 'quotaValue': '20'}]}} "
    "https://ai.google.dev/gemini-api/docs/rate-limits"
)


class _NoisyGeminiError(_FakeGeminiError):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self._noise = _UPSTREAM_NOISE

    def __str__(self) -> str:  # pragma: no cover - mesaj icerigi test ediliyor
        return f"{self.code} {self._noise}"


@pytest.mark.parametrize(
    ("code", "status"),
    [(429, 429), (500, 502)],
)
def test_upstream_error_text_is_not_forwarded_to_the_client(code, status, monkeypatch):
    local = _client_with_error(_NoisyGeminiError(code), monkeypatch)

    response = local.post("/agent/ask", json={"question": "soru"})

    assert response.status_code == status
    message = response.json()["detail"]["message"]
    for leak in ("RESOURCE_EXHAUSTED", "quotaMetric", "https://", "generativelanguage"):
        assert leak not in message, f"upstream ayrinti sizdi: {leak}"
    assert len(message) <= 200
    # Mesaj yine de kullaniciya ne oldugunu soylemeli.
    assert message.strip()


def test_upstream_error_detail_is_written_to_the_server_log(monkeypatch, caplog):
    """Ayrinti kaybolmamali: istemciden gizlenir, logda tutulur."""
    local = _client_with_error(_NoisyGeminiError(429), monkeypatch)

    with caplog.at_level("WARNING", logger="rag_tr.api.agent_routes"):
        local.post("/agent/ask", json={"question": "soru"})

    assert any("RESOURCE_EXHAUSTED" in record.getMessage() for record in caplog.records)


def test_unrelated_errors_are_not_labelled_as_quota(monkeypatch):
    """Alakasiz bir hata kota hatasi gibi gosterilmemeli."""
    local = _client_with_error(RuntimeError("beklenmeyen"), monkeypatch)

    response = local.post("/agent/ask", json={"question": "soru"})

    assert response.status_code == 500
    assert ErrorCode.QUOTA_EXHAUSTED.value not in response.text


# --- agent kurulamadiginda ---


def test_missing_gemini_key_makes_only_the_agent_endpoint_unavailable(monkeypatch):
    from rag_tr.agent.config import AgentSettings

    monkeypatch.setattr(deps, "AgentSettings", lambda: AgentSettings(gemini_api_key=None))
    local = TestClient(create_app(service=_StubService()))

    response = local.post("/agent/ask", json={"question": "soru"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == ErrorCode.AGENT_UNAVAILABLE.value


def test_creating_the_app_does_not_build_an_agent():
    """create_app() Gemini key'i veya ag erisimi gerektirmemeli."""
    app = create_app(service=_StubService())

    assert app.state.agent is None
    assert app.state.agent_factory is None


# --- bos korpus ---


def test_empty_corpus_returns_a_clear_error_without_running_the_agent():
    """Hic dokuman ingest edilmemisse agent hic calistirilmaz: bosa Gemini
    cagrisi yapilmaz ve istemci bunu "sistem bozuk" ile karistirmaz."""
    agent = FakeAgent()
    local = TestClient(create_app(service=_StubService(chunk_count=0), agent=agent))

    response = local.post("/agent/ask", json={"question": "Başkent neresi?"})

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == ErrorCode.CORPUS_EMPTY.value
    assert "ingest" in detail["message"].lower()
    assert agent.questions == [], "agent hic cagrilmamali"


def test_non_empty_corpus_runs_the_agent():
    agent = FakeAgent()
    local = TestClient(create_app(service=_StubService(chunk_count=1), agent=agent))

    assert local.post("/agent/ask", json={"question": "soru"}).status_code == 200
    assert agent.questions == ["soru"]
