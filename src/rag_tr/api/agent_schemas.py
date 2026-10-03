"""Agent HTTP sozlesmesi.

Web istemcisinin gordugu tek yuzey budur. Burada yalnizca AgentResult'in
gozlemlenebilir alanlari yer alir; RAGService, VectorStore, BM25Index,
ChromaDB, EmbeddingModel veya Gemini SDK tiplerinden hicbiri disa acilmaz.
Trace yalnizca kapali bir enum olan aksiyon etiketi ve kisa bir ozet detay
tasir -- dusunce zinciri (chain-of-thought) hicbir alanda bulunmaz.
"""

from pydantic import BaseModel, Field, field_validator

from rag_tr.agent.contracts import AgentAction, AgentStatus, ToolStatus
from rag_tr.i18n import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES, normalize_language


class AgentAskRequest(BaseModel):
    question: str = Field(min_length=1)
    # ge=1: top_k=0 sessizce varsayilana dusmek yerine acik bir dogrulama
    # hatasi verir (mevcut /query sozlesmesiyle ayni kural).
    top_k: int | None = Field(default=None, ge=1)
    # Cevabin dili. Istemci gondermezse korpusun dili kullanilir.
    # Retrieval bu alandan etkilenmez: arama sorgusu her zaman korpus
    # dilinde uretilir, yalnizca cevap ve gerekceler bu dile cevrilir.
    language: str = Field(default=DEFAULT_LANGUAGE)

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str) -> str:
        """Desteklenen bir dil koduna indirger.

        `en-US` gibi bolgeli kodlar kabul edilip dil kismina indirgenir;
        hic desteklenmeyen bir dil ise sessizce varsayilana dusmek yerine
        acik bir dogrulama hatasi verir. Sessiz dusus, arayuzde bir yazim
        hatasini gorunmez kilardi."""
        raw = (value or DEFAULT_LANGUAGE).strip()
        normalized = normalize_language(raw)
        base = raw.replace("_", "-").split("-")[0].lower()
        if base and base != normalized:
            raise ValueError(
                f"desteklenmeyen dil: {raw!r}; desteklenenler: {', '.join(SUPPORTED_LANGUAGES)}"
            )
        return normalized


class AgentSourceItem(BaseModel):
    """Retrieve edilen tek bir pasaj (rag_tr.contracts.Passage'in public hali)."""

    chunk_id: str
    source_file: str
    page_number: int | None
    text: str
    rank: int
    score: float


class AgentStepItem(BaseModel):
    """Guvenli, ust duzey aksiyon kaydi."""

    action: AgentAction
    detail: str


class AgentToolCallItem(BaseModel):
    tool: str
    query: str
    top_k: int
    source_file: str | None
    status: ToolStatus
    passages_found: int


class AgentAskResponse(BaseModel):
    answer: str
    status: AgentStatus
    sources: list[AgentSourceItem]
    steps: list[AgentStepItem]
    tool_calls: list[AgentToolCallItem]
