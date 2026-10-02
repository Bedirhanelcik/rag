"""Agent davranisinin promptevals ile degerlendirilmesi.

Bu paket yalnizca bir adapter'dir: degerlendirme mantigi (checker'lar ve LLM
judge) promptevals projesinden oldugu gibi kullanilir, burada yeniden
yazilmaz. promptevals de RAG veya agent ici hakkinda hicbir sey bilmez --
kendisine yalnizca gozlemlenebilir metin ve assertion config'i verilir.
"""

from rag_tr.agent.eval.cases import (
    AgentCase,
    AgentSuite,
    JudgeSpec,
    compile_assertions,
    load_suite,
)
from rag_tr.agent.eval.observation import render_observation
from rag_tr.agent.eval.runner import (
    AgentCaseResult,
    AgentEvalSummary,
    evaluate_agent_case,
    run_agent_eval,
)

__all__ = [
    "AgentCase",
    "AgentCaseResult",
    "AgentEvalSummary",
    "AgentSuite",
    "JudgeSpec",
    "compile_assertions",
    "evaluate_agent_case",
    "load_suite",
    "render_observation",
    "run_agent_eval",
]
