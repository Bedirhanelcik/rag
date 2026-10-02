"""Agent katmaninin tiplenmis sozlesmesi.

Bu modul agent ile cagiran taraf (ileride web UI) arasindaki siniri tanimlar.
Trace yalnizca ust duzey, guvenli aksiyon etiketleri tasir -- modelin dusunce
zinciri (chain-of-thought) hicbir yerde saklanmaz veya disa verilmez.
"""

from dataclasses import dataclass, field
from enum import Enum

from rag_tr.contracts import Passage


class AgentStatus(str, Enum):
    """Bir agent kosusunun makine-okunur sonucu."""

    ANSWERED = "answered"
    ANSWERED_WITHOUT_RETRIEVAL = "answered_without_retrieval"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    TOOL_FAILURE = "tool_failure"


class AgentAction(str, Enum):
    """Trace'te gorunebilecek aksiyonlarin tamami.

    Kasitli olarak kaba taneli: "neyi yapti" bilgisini verir, "neden boyle
    dusundu" bilgisini vermez."""

    ASSESSED_QUESTION = "assessed_question"
    SEARCHED_RAG = "searched_rag"
    REFINED_QUERY = "refined_query"
    ANSWERED_FROM_CONTEXT = "answered_from_context"
    ANSWERED_WITHOUT_RETRIEVAL = "answered_without_retrieval"
    DECLARED_INSUFFICIENT_CONTEXT = "declared_insufficient_context"
    TOOL_FAILED = "tool_failed"


class ToolStatus(str, Enum):
    """Tool cagrisinin sonucu. Tool'lar istisna firlatmaz; agent bu durumlar
    uzerinde muhakeme eder."""

    OK = "ok"
    EMPTY = "empty"
    FAILED = "failed"


@dataclass(frozen=True)
class SearchOutcome:
    status: ToolStatus
    passages: list[Passage] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class ToolCall:
    """UI ve denetim icin tek bir tool cagrisinin kaydi."""

    tool: str
    query: str
    top_k: int
    source_file: str | None
    status: ToolStatus
    passages_found: int


@dataclass(frozen=True)
class Step:
    action: AgentAction
    detail: str = ""


@dataclass(frozen=True)
class RetrievalDecision:
    """LLM'in "bu soru icin dokuman aramasi gerekiyor mu" karari."""

    needs_retrieval: bool
    search_query: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class PassageAssessment:
    """LLM'in "bulunan pasajlar soruyu cevaplamaya yetiyor mu" karari."""

    sufficient: bool
    refined_query: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class AgentResult:
    answer: str
    status: AgentStatus
    sources: list[Passage] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
