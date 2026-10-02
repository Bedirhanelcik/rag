"""Agent degerlendirme kosucusu.

Agent'i bir kez kosturur, gozlemlenebilir sonucu promptevals'in kendi
checker'larina verir. promptevals tarafinda RAG veya agent ici hakkinda hicbir
bilgi yoktur; buradan ona yalnizca metin ve assertion config'i gider.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable

from promptevals.assertions import SYNC_CHECKERS
from promptevals.judge import run_llm_judge
from promptevals.models import AssertionResult

from rag_tr.agent.contracts import AgentResult
from rag_tr.agent.eval.cases import AgentCase, compile_assertions
from rag_tr.agent.eval.observation import render_observation


@dataclass
class AgentCaseResult:
    case_id: str
    question: str
    result: AgentResult
    observation: str
    assertion_results: list[AssertionResult]
    latency_ms: float
    llm_calls: int | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(a.passed for a in self.assertion_results)


@dataclass
class AgentEvalSummary:
    results: list[AgentCaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def avg_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.results) / self.total if self.total else 0.0


async def evaluate_agent_case(
    agent,
    case: AgentCase,
    *,
    judge_client=None,
    call_count: Callable[[], int] | None = None,
) -> AgentCaseResult:
    """Tek bir vakayi kosturur ve degerlendirir.

    `agent` yalnizca `run(question) -> AgentResult` sozlesmesi uzerinden
    kullanilir; hangi LLM saglayicisini kullandigi onemli degildir."""
    calls_before = call_count() if call_count else None

    start = time.perf_counter()
    try:
        result = agent.run(case.question)
    except Exception as exc:  # noqa: BLE001 - vaka hatasi olarak raporlanir, kosu durmaz
        latency_ms = (time.perf_counter() - start) * 1000
        return AgentCaseResult(
            case_id=case.id,
            question=case.question,
            result=None,
            observation="",
            assertion_results=[],
            latency_ms=latency_ms,
            llm_calls=None,
            error=str(exc),
        )
    latency_ms = (time.perf_counter() - start) * 1000

    observation = render_observation(result)
    targets = {"observation": observation, "answer": result.answer}

    assertion_results: list[AssertionResult] = []
    for compiled in compile_assertions(case):
        assertion = compiled.assertion
        output = targets[compiled.target]
        try:
            if assertion.type == "llm_judge":
                # Anthropic client'i verilmez: degerlendirme Gemini free tier
                # uzerinden calisir.
                check = await run_llm_judge(None, judge_client, assertion, output)
            else:
                check = SYNC_CHECKERS[assertion.type](assertion, output)
        except Exception as exc:  # noqa: BLE001 - basarisiz assertion olarak raporlanir
            check = AssertionResult(
                type=assertion.type, passed=False, detail=f"assertion hata verdi: {exc}"
            )
        assertion_results.append(check)

    return AgentCaseResult(
        case_id=case.id,
        question=case.question,
        result=result,
        observation=observation,
        assertion_results=assertion_results,
        latency_ms=latency_ms,
        llm_calls=(call_count() - calls_before) if calls_before is not None else None,
    )


async def run_agent_eval(
    agent,
    cases: list[AgentCase],
    *,
    judge_client=None,
    call_count: Callable[[], int] | None = None,
    delay_seconds: float = 0.0,
    sleep=asyncio.sleep,
) -> AgentEvalSummary:
    """Vakalari sirayla kosturur.

    Kasitli olarak seri. `delay_seconds` vakalar arasina bekleme koyar: Gemini
    free tier dakikada 5 istekle sinirli ve her vaka 2-3 istek harciyor, bu
    yuzden canli kosuda arka arkaya gitmek kotayi tuketiyor. Testlerde
    varsayilan 0'dir, dolayisiyla bekleme olmaz."""
    results = []
    for index, case in enumerate(cases):
        if index and delay_seconds > 0:
            await sleep(delay_seconds)
        results.append(
            await evaluate_agent_case(
                agent, case, judge_client=judge_client, call_count=call_count
            )
        )
    return AgentEvalSummary(results=results)
