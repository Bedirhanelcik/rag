"""Agent degerlendirme vakalari ve bunlarin promptevals assertion'larina derlenmesi.

Burada hicbir checker yeniden yazilmaz. YAML'daki okunabilir beklenti alanlari
(`expect_status`, `expect_source`, ...) promptevals'in kendi assertion
config'lerine derlenir ve kontrolu promptevals'in `SYNC_CHECKERS`/`run_llm_judge`
fonksiyonlari yapar. Tek sorumlulugu: "agent davranisi" ile "string assertion"
arasindaki ceviri.
"""

import re
from pathlib import Path
from typing import Literal

import yaml
from promptevals.config import (
    AssertionConfig,
    ContainsAssertion,
    LlmJudgeAssertion,
    MaxLengthAssertion,
    NotContainsAssertion,
    RegexAssertion,
)
from pydantic import BaseModel, Field

# Assertion'in hangi metne uygulanacagi: tum gozlem belgesi mi, yalnizca cevap mi.
Target = Literal["observation", "answer"]


class JudgeSpec(BaseModel):
    rubric: str
    threshold: int = 3
    # Gemini free tier; promptevals judge_model'in isim alanina gore backend secer.
    judge_model: str = "gemini-2.5-flash"


class AgentCase(BaseModel):
    """Tek bir agent davranis vakasi."""

    id: str
    question: str

    expect_status: str | list[str] | None = None
    expect_tool: str | None = None
    expect_no_tool: bool = False
    expect_tool_calls: int | None = None
    expect_action: str | list[str] | None = None
    expect_source: str | list[str] | None = None

    answer_contains: list[str] = Field(default_factory=list)
    answer_not_contains: list[str] = Field(default_factory=list)
    answer_regex: str | None = None
    answer_max_length: int | None = None

    # Yalnizca gercekten anlamsal degerlendirme gerektiginde; her vakaya
    # LLM judge cagrisi harcanmaz.
    judge: JudgeSpec | None = None


class AgentSuite(BaseModel):
    cases: list[AgentCase]


class CompiledAssertion(BaseModel):
    assertion: AssertionConfig
    target: Target


def load_suite(path: str | Path) -> AgentSuite:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AgentSuite.model_validate(raw)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _line_contains(line_key: str, token: str) -> RegexAssertion:
    """`<line_key>: ... [token] ...` kalibini arar -- degerin yalnizca ilgili
    satirda gecmesini garanti eder, cevap metnindeki tesadufi eslesmeleri sayMAZ."""
    return RegexAssertion(
        type="regex", pattern=rf"{re.escape(line_key)}:[^\n]*\[{re.escape(token)}\]"
    )


def compile_assertions(case: AgentCase) -> list[CompiledAssertion]:
    """Vakanin beklentilerini promptevals assertion config'lerine cevirir."""
    compiled: list[CompiledAssertion] = []

    def add(assertion: AssertionConfig, target: Target = "observation") -> None:
        compiled.append(CompiledAssertion(assertion=assertion, target=target))

    statuses = _as_list(case.expect_status)
    if len(statuses) == 1:
        add(ContainsAssertion(type="contains", value=f"status: [{statuses[0]}]"))
    elif statuses:
        alternatives = "|".join(re.escape(s) for s in statuses)
        add(RegexAssertion(type="regex", pattern=rf"status: \[(?:{alternatives})\]"))

    if case.expect_tool:
        add(_line_contains("tools_used", case.expect_tool))

    if case.expect_no_tool:
        add(ContainsAssertion(type="contains", value="tool_calls: 0"))

    if case.expect_tool_calls is not None:
        add(RegexAssertion(type="regex", pattern=rf"tool_calls: {case.expect_tool_calls}(?!\d)"))

    for action in _as_list(case.expect_action):
        add(_line_contains("actions", action))

    for source in _as_list(case.expect_source):
        add(_line_contains("sources", source))

    for value in case.answer_contains:
        add(ContainsAssertion(type="contains", value=value), target="answer")

    for value in case.answer_not_contains:
        add(NotContainsAssertion(type="not_contains", value=value), target="answer")

    if case.answer_regex:
        add(RegexAssertion(type="regex", pattern=case.answer_regex), target="answer")

    if case.answer_max_length is not None:
        add(MaxLengthAssertion(type="max_length", value=case.answer_max_length), target="answer")

    if case.judge is not None:
        add(
            LlmJudgeAssertion(
                type="llm_judge",
                rubric=case.judge.rubric,
                threshold=case.judge.threshold,
                judge_model=case.judge.judge_model,
            ),
            target="answer",
        )

    return compiled
