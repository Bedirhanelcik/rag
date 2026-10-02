"""Agent + Gemini icin tek seferlik canli dogrulama.

Calistirma:
    uv run --no-editable python scripts/agent_smoke.py
    uv run --no-editable python scripts/agent_smoke.py "Kendi sorunuz?"

Gemini'ye mumkun olan en az sayida istek gider (karar + degerlendirme + sentez;
ikinci arama gerekirse bir degerlendirme daha). Retrieval tamamen localdir ve
Anthropic API hic cagrilmaz. API key hicbir zaman yazdirilmaz.
"""

import sys
from pathlib import Path

from rag_tr.agent.config import AgentSettings
from rag_tr.agent.gemini import GeminiAgentLLM
from rag_tr.agent.loop import ResearchAgent
from rag_tr.agent.tools import RagSearchTool
from rag_tr.config import Settings
from rag_tr.service import RAGService

DEFAULT_QUESTION = "Türkiye'nin en yüksek dağı hangisidir ve yüksekliği kaç metredir?"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    agent_settings = AgentSettings()
    if not agent_settings.gemini_api_key:
        print("ATLANDI: GEMINI_API_KEY .env dosyasinda tanimli degil.")
        return 1

    settings = Settings()
    service = RAGService(settings)

    if service.vector_store.count() == 0:
        samples = sorted((PROJECT_ROOT / "data" / "sample").glob("*.md"))
        ingest = service.ingest_files(samples)
        print(f"ingest : {ingest.ingested_files} -> {ingest.chunk_count} chunk")

    print(f"korpus : {service.vector_store.count()} chunk / BM25 {service.bm25_index.size()}")

    llm = GeminiAgentLLM.from_env(agent_settings)
    agent = ResearchAgent(llm=llm, search_tool=RagSearchTool(service))

    print(f"model  : {llm.model}")
    print(f"soru   : {question}\n")

    result = agent.run(question)

    print(f"status : {result.status.value}")
    print(f"cevap  : {result.answer}\n")
    print("kaynaklar:")
    for source in result.sources:
        print(f"  - {source.chunk_id} (rank {source.rank}, skor {source.score:.4f})")
    print("\nizleme (action trace):")
    for i, step in enumerate(result.steps, start=1):
        print(f"  {i}. {step.action.value}: {step.detail}")
    print("\ntool cagrilari:")
    for call in result.tool_calls:
        print(f"  - {call.tool}(query={call.query!r}, top_k={call.top_k}) -> "
              f"{call.status.value}, {call.passages_found} pasaj")
    print(f"\ngemini cagri sayisi : {llm.call_count}")
    print(f"rag kullanildi      : {bool(result.tool_calls)}")
    print(f"ikinci arama gerekti: {len(result.tool_calls) > 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
