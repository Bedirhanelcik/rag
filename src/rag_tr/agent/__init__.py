"""RAG uzerine kurulu agent katmani.

Agent yalnizca rag_tr.contracts tiplerini ve bir arama tool'unu gorur;
VectorStore / BM25Index / ChromaDB / EmbeddingModel detaylari disa acilmaz.
"""

from rag_tr.agent.contracts import (
    AgentAction,
    AgentResult,
    AgentStatus,
    PassageAssessment,
    RetrievalDecision,
    SearchOutcome,
    Step,
    ToolCall,
    ToolStatus,
)
from rag_tr.agent.loop import AgentLLM, ResearchAgent
from rag_tr.agent.tools import RagSearchTool, SearchTool

__all__ = [
    "AgentAction",
    "AgentLLM",
    "AgentResult",
    "AgentStatus",
    "PassageAssessment",
    "RagSearchTool",
    "ResearchAgent",
    "RetrievalDecision",
    "SearchOutcome",
    "SearchTool",
    "Step",
    "ToolCall",
    "ToolStatus",
]
