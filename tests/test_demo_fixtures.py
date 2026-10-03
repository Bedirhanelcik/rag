"""Demo fixture dosyasinin dogrulama testleri.

Hicbir canli API cagrisi yapilmaz; bir test bunu LLM istemcilerini patlayacak
sekilde degistirerek acikca kanitlar.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_tr.agent.contracts import AgentAction, AgentStatus, ToolStatus
from rag_tr.api.agent_schemas import AgentAskResponse
from rag_tr.demo import DemoFixtureFile, load_fixtures

FIXTURES_PATH = Path(__file__).resolve().parent.parent / "demo" / "fixtures.json"

EXPECTED_FIXTURE_COUNT = 5

# Dosyada bulunmasina izin verilen anahtarlarin tamami. Gizli bir reasoning
# alani eklenirse bu allowlist onu yakalar (Pydantic tarafinda da
# extra="forbid" ikinci bir savunma katmani olarak duruyor).
ALLOWED_KEYS = {
    "root": {"mode", "note", "fixtures"},
    "fixture": {
        "id",
        "question",
        "synthetic",
        "provenance",
        "answer",
        "status",
        "sources",
        "steps",
        "tool_calls",
    },
    "provenance": {"source", "note", "reconstructed_fields"},
    "source": {"chunk_id", "source_file", "page_number", "text", "rank", "score"},
    "step": {"action", "detail"},
    "tool_call": {"tool", "query", "top_k", "source_file", "status", "passages_found"},
}

# Gizli muhakeme tasiyabilecek alan adlari.
FORBIDDEN_KEY_FRAGMENTS = (
    "reasoning",
    "thought",
    "chain_of_thought",
    "cot",
    "scratchpad",
    "deliberation",
    "system_prompt",
    "prompt",
    "raw_response",
)


@pytest.fixture(scope="module")
def raw() -> dict:
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def suite() -> DemoFixtureFile:
    return load_fixtures(FIXTURES_PATH)


# --- parse / dogrulama ---


def test_fixture_file_parses_as_json(raw):
    assert isinstance(raw, dict)


def test_fixture_file_validates_against_the_schema(suite):
    assert isinstance(suite, DemoFixtureFile)


def test_file_is_marked_as_demo_data(suite, raw):
    assert suite.mode == "demo"
    assert raw["mode"] == "demo"
    assert suite.note.strip()


def test_there_are_exactly_five_fixtures(suite):
    assert len(suite.fixtures) == EXPECTED_FIXTURE_COUNT


def test_fixture_ids_are_unique(suite):
    ids = [fixture.id for fixture in suite.fixtures]
    assert len(set(ids)) == len(ids)


# --- zorunlu alanlar ve sema uyumu ---


def test_every_fixture_carries_the_required_fields(raw):
    for fixture in raw["fixtures"]:
        for field in ("question", "answer", "status", "sources", "steps", "tool_calls"):
            assert field in fixture, f"{fixture.get('id')} icinde {field} yok"
        assert "synthetic" in fixture
        assert isinstance(fixture["synthetic"], bool)


def test_every_status_is_a_valid_agent_status(suite):
    valid = {status for status in AgentStatus}
    for fixture in suite.fixtures:
        assert fixture.status in valid


def test_sources_steps_and_tool_calls_conform_to_the_schema(suite):
    for fixture in suite.fixtures:
        for source in fixture.sources:
            assert source.chunk_id.startswith(f"{source.source_file}::")
            assert source.rank >= 1
            assert isinstance(source.score, float)
            assert source.text.strip()
        assert [s.rank for s in fixture.sources] == list(range(1, len(fixture.sources) + 1))

        for step in fixture.steps:
            assert isinstance(step.action, AgentAction)
            assert len(step.detail) <= 200

        for call in fixture.tool_calls:
            assert call.tool == "rag_search"
            assert isinstance(call.status, ToolStatus)
            assert call.top_k >= 1
            assert call.passages_found >= 0


def test_every_fixture_converts_to_the_live_response_schema(suite):
    """Demo verisi ile canli /agent/ask yaniti ayni sekilde olmali."""
    for fixture in suite.fixtures:
        response = fixture.to_response()
        assert isinstance(response, AgentAskResponse)
        assert set(response.model_dump()) == {
            "answer",
            "status",
            "sources",
            "steps",
            "tool_calls",
        }


def test_statuses_are_internally_consistent(suite):
    for fixture in suite.fixtures:
        if fixture.status is AgentStatus.ANSWERED_WITHOUT_RETRIEVAL:
            assert fixture.tool_calls == [], f"{fixture.id}: arama yapilmamis olmali"
            assert fixture.sources == []
        if fixture.status is AgentStatus.INSUFFICIENT_CONTEXT:
            assert fixture.sources == [], f"{fixture.id}: kaynak gosterilmemeli"
        if fixture.status is AgentStatus.ANSWERED:
            assert fixture.sources, f"{fixture.id}: cevap kaynaksiz olamaz"


# --- demo/synthetic isaretlemesi ---


def test_exactly_one_fixture_is_synthetic(suite):
    synthetic = [fixture for fixture in suite.fixtures if fixture.synthetic]

    assert len(synthetic) == 1
    assert synthetic[0].id == "synthetic_query_refinement"


def test_non_synthetic_fixtures_declare_a_live_provenance(suite):
    for fixture in suite.fixtures:
        if not fixture.synthetic:
            assert fixture.provenance.source in {"live_smoke_test", "live_eval_suite"}
        else:
            assert fixture.provenance.source == "handcrafted"


def test_provenance_records_which_fields_were_reconstructed(suite):
    for fixture in suite.fixtures:
        assert fixture.provenance.note.strip()
        assert isinstance(fixture.provenance.reconstructed_fields, list)
    synthetic = suite.by_id("synthetic_query_refinement")
    assert synthetic.provenance.reconstructed_fields, "kurgulanan alanlar belirtilmeli"


# --- synthetic fixture gercekten ikinci aramayi temsil ediyor ---


def test_synthetic_fixture_represents_a_real_second_search(suite):
    fixture = suite.by_id("synthetic_query_refinement")

    assert fixture.synthetic is True
    assert len(fixture.tool_calls) == 2, "iki ayri rag_search olmali"

    actions = [step.action for step in fixture.steps]
    assert actions == [
        AgentAction.ASSESSED_QUESTION,
        AgentAction.SEARCHED_RAG,
        AgentAction.REFINED_QUERY,
        AgentAction.SEARCHED_RAG,
        AgentAction.ANSWERED_FROM_CONTEXT,
    ]

    first, second = fixture.tool_calls
    assert first.query != second.query, "ikinci arama yeniden formullenmis sorguyu kullanmali"
    assert first.status is ToolStatus.EMPTY and first.passages_found == 0
    assert second.status is ToolStatus.OK and second.passages_found >= 1
    assert fixture.status is AgentStatus.ANSWERED
    assert fixture.sources


def test_no_other_fixture_is_presented_as_a_handcrafted_success(suite):
    for fixture in suite.fixtures:
        if fixture.id != "synthetic_query_refinement":
            assert not fixture.synthetic


# --- gizli muhakeme alani yok ---


def _walk_keys(node, kind: str, found: list[tuple[str, str]]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            found.append((kind, key))
            if kind == "root" and key == "fixtures":
                for item in value:
                    _walk_keys(item, "fixture", found)
            elif kind == "fixture" and key == "provenance":
                _walk_keys(value, "provenance", found)
            elif kind == "fixture" and key == "sources":
                for item in value:
                    _walk_keys(item, "source", found)
            elif kind == "fixture" and key == "steps":
                for item in value:
                    _walk_keys(item, "step", found)
            elif kind == "fixture" and key == "tool_calls":
                for item in value:
                    _walk_keys(item, "tool_call", found)


def test_only_allowlisted_keys_appear_anywhere_in_the_file(raw):
    found: list[tuple[str, str]] = []
    _walk_keys(raw, "root", found)

    assert found, "anahtar bulunamadi"
    for kind, key in found:
        assert key in ALLOWED_KEYS[kind], f"{kind} icinde beklenmeyen anahtar: {key}"


def test_no_key_hints_at_hidden_reasoning(raw):
    found: list[tuple[str, str]] = []
    _walk_keys(raw, "root", found)

    for _kind, key in found:
        lowered = key.lower()
        for fragment in FORBIDDEN_KEY_FRAGMENTS:
            assert fragment not in lowered, f"gizli muhakeme alani olabilir: {key}"


def test_schema_rejects_an_injected_reasoning_field():
    """extra='forbid' ikinci savunma katmani: bilinmeyen alan reddedilmeli."""
    payload = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    payload["fixtures"][0]["reasoning"] = "gizli dusunce zinciri"

    with pytest.raises(ValidationError):
        DemoFixtureFile.model_validate(payload)


def test_schema_rejects_a_fixture_without_the_synthetic_flag():
    payload = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    del payload["fixtures"][0]["synthetic"]

    with pytest.raises(ValidationError):
        DemoFixtureFile.model_validate(payload)


# --- canli API gerektirmiyor ---


def test_loading_fixtures_never_constructs_an_llm_client(monkeypatch):
    """Demo yolunun Gemini/Anthropic istemcisi kurmadigini kanitlar."""
    import google.genai

    def _explode(*args, **kwargs):
        raise AssertionError("demo yolunda LLM istemcisi kurulmamali")

    monkeypatch.setattr(google.genai, "Client", _explode)

    # anthropic artik opsiyonel bir bagimlilik (legacy extra); kuruluysa onu da
    # patlat, kurulu degilse zaten kurulma ihtimali yok.
    try:
        import anthropic
    except ImportError:
        pass
    else:
        monkeypatch.setattr(anthropic, "Anthropic", _explode)

    suite = load_fixtures(FIXTURES_PATH)
    for fixture in suite.fixtures:
        fixture.to_response()

    assert len(suite.fixtures) == EXPECTED_FIXTURE_COUNT


# --- arama yardimcilari ---


def test_lookup_by_id_and_by_question(suite):
    fixture = suite.by_id("factual_rag_answer")
    assert fixture is not None
    # Bosluk/yeni satir farki onemsenmemeli.
    padded = "  " + fixture.question + " \n"
    assert suite.by_question(padded) is fixture
    assert suite.by_id("olmayan") is None
    assert suite.by_question("hiç sorulmamış bir soru") is None


def test_lookup_is_case_insensitive(suite):
    """Not: Python Turkce locale kullanmadigi icin noktasiz 'i' uzerinden
    buyuk-kucuk donusu tersine cevrilemez ("dagi".upper().casefold() == "dagi"
    degil "dagi" -> "DAGI" -> "dagi"); bu yuzden test noktasiz i icermeyen bir
    soruyla yapiliyor. Ayni tuzak P3'te arayuz eslestirmesi icin de gecerli."""
    fixture = suite.by_id("synthetic_query_refinement")

    assert suite.by_question(fixture.question.upper()) is fixture
