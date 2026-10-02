"""Agent davranis degerlendirmesini canli Gemini free tier ile kosturur (opt-in).

Calistirma:
    uv run --no-editable python scripts/agent_eval.py
    uv run --no-editable python scripts/agent_eval.py eval/agent_suite.yaml

Maliyet: Gemini free tier, vaka sayisi kasitli olarak kucuk. Vakalarin yalnizca
birinde LLM judge kullanilir; geri kalan tum kontroller deterministik ve
ucretsizdir. API key hicbir zaman yazdirilmaz.
"""

import asyncio
import sys
from pathlib import Path

from google import genai
from promptevals.cost import estimate_gemini_cost_usd

from rag_tr.agent.config import AgentSettings
from rag_tr.agent.eval import load_suite, run_agent_eval
from rag_tr.agent.gemini import GeminiAgentLLM
from rag_tr.agent.loop import ResearchAgent
from rag_tr.agent.tools import RagSearchTool
from rag_tr.config import Settings
from rag_tr.service import RAGService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUITE = PROJECT_ROOT / "eval" / "agent_suite.yaml"
# Gemini free tier: 5 istek/dakika. Her vaka 2-3 istek harciyor, bu yuzden
# vakalar arasina bekleme konuyor -- kosu yavas ama kotayi tuketmiyor.
CASE_DELAY_SECONDS = 45.0


async def main() -> int:
    suite_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SUITE

    agent_settings = AgentSettings()
    if not agent_settings.gemini_api_key:
        print("ATLANDI: GEMINI_API_KEY tanimli degil.")
        return 1

    suite = load_suite(suite_path)
    service = RAGService(Settings())
    if service.vector_store.count() == 0:
        ingest = service.ingest_files(sorted((PROJECT_ROOT / "data" / "sample").glob("*.md")))
        print(f"ingest : {ingest.ingested_files} -> {ingest.chunk_count} chunk")

    # Tek client: hem agent'in LLM'i hem de judge icin kullanilir.
    client = genai.Client(api_key=agent_settings.gemini_api_key)
    llm = GeminiAgentLLM(client, model=agent_settings.gemini_agent_model)
    agent = ResearchAgent(llm=llm, search_tool=RagSearchTool(service))

    print(f"suite  : {suite_path.name} ({len(suite.cases)} vaka)")
    print(f"model  : {llm.model}")
    print(f"korpus : {service.vector_store.count()} chunk\n")

    summary = await run_agent_eval(
        agent,
        suite.cases,
        judge_client=client,
        call_count=lambda: llm.call_count,
        delay_seconds=CASE_DELAY_SECONDS,
    )

    for case in summary.results:
        mark = "PASS" if case.passed else "FAIL"
        status = case.result.status.value if case.result else "hata"
        print(f"[{mark}] {case.case_id:34} status={status:28} "
              f"{case.latency_ms:7.0f}ms  gemini={case.llm_calls}")
        if case.error:
            print(f"        hata: {case.error}")
        for assertion in case.assertion_results:
            if not assertion.passed:
                print(f"        - {assertion.type}: {assertion.detail}")

    free = estimate_gemini_cost_usd(llm.model, llm.input_tokens, llm.output_tokens, True)
    paid = estimate_gemini_cost_usd(llm.model, llm.input_tokens, llm.output_tokens, False)

    print(f"\ntoplam vaka    : {summary.total}")
    print(f"gecen          : {summary.passed}")
    print(f"basarisiz      : {summary.failed}")
    print(f"gemini cagrisi : {llm.call_count} (agent) + judge cagrilari")
    print(f"token (agent)  : girdi {llm.input_tokens}, cikti {llm.output_tokens}")
    print(f"maliyet        : ${free if free is not None else 0.0:.4f} (free tier)"
          f" | paid tier esdegeri: ${paid:.4f}" if paid is not None else "")
    print(f"ortalama sure  : {summary.avg_latency_ms:.0f} ms")
    return 0 if summary.failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
