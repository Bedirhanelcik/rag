"""Agent HTTP sozlesmesi.

Web istemcisinin gordugu tek yuzey budur. Burada yalnizca AgentResult'in
gozlemlenebilir alanlari yer alir; RAGService, VectorStore, BM25Index,
ChromaDB, EmbeddingModel veya Gemini SDK tiplerinden hicbiri disa acilmaz.
Trace yalnizca kapali bir enum olan aksiyon etiketi ve kisa bir ozet detay
tasir -- dusunce zinciri (chain-of-thought) hicbir alanda bulunmaz.
"""

from pydantic import BaseModel, Field

from rag_tr.agent.contracts import AgentAction, AgentStatus, ToolStatus


class AgentAskRequest(BaseModel):
    question: str = Field(min_length=1)
    # ge=1: top_k=0 sessizce varsayilana dusmek yerine acik bir dogrulama
    # hatasi verir (mevcut /query sozlesmesiyle ayni kural).
    top_k: int | None = Field(default=None, ge=1)


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
