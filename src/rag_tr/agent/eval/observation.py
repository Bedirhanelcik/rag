"""AgentResult'in gozlemlenebilir yuzeyini deterministik bir metne cevirir.

Degerlendirme bu metin uzerinde yapilir; boylece promptevals'in mevcut
assertion checker'lari (contains/not_contains/regex/max_length) hicbir
degisiklige gerek kalmadan agent davranisini olcebilir. Metin satir bazli ve
siralamasi sabittir; enum benzeri degerler koseli parantez icinde yazilir ki
`[answered]` ile `[answered_without_retrieval]` birbirine karismasin.
"""

from rag_tr.agent.contracts import AgentResult


def _bracketed(values) -> str:
    return "".join(f"[{value}]" for value in values)


def render_observation(result: AgentResult) -> str:
    """Agent'in disaridan gozlemlenebilen davranisini metne dokar.

    VectorStore / BM25Index / ChromaDB / EmbeddingModel / Gemini SDK
    detaylarindan hicbiri buraya girmez."""
    answer_one_line = " ".join(result.answer.split())
    lines = [
        f"status: [{result.status.value}]",
        f"tool_calls: {len(result.tool_calls)}",
        f"tools_used: {_bracketed(dict.fromkeys(call.tool for call in result.tool_calls))}",
        f"search_queries: {_bracketed(call.query for call in result.tool_calls)}",
        f"actions: {_bracketed(step.action.value for step in result.steps)}",
        f"sources: {_bracketed(source.chunk_id for source in result.sources)}",
        f"source_files: {_bracketed(dict.fromkeys(s.source_file for s in result.sources))}",
        f"answer: {answer_one_line}",
    ]
    return "\n".join(lines)
