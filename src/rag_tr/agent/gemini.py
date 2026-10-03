"""AgentLLM'in Gemini (Google AI Studio free tier) implementasyonu.

ResearchAgent bu modulu tanimaz: yalnizca dort metotluk AgentLLM yuzeyini
cagirir. Saglayici degistirmek icin dongude tek satir degismesi gerekmez.

Kararlar structured output (response_schema) ile alinir, serbest metin
ayristirilmaz; sicaklik 0 ve ayristirma hatalarinda guvenli tarafa dusen
varsayilanlar sayesinde davranis olabildigince deterministiktir.
"""

import time

from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

from rag_tr.agent.config import AgentSettings
from rag_tr.agent.contracts import PassageAssessment, RetrievalDecision
from rag_tr.contracts import Passage
from rag_tr.i18n import (
    CORPUS_LANGUAGE_NAME,
    DEFAULT_LANGUAGE,
    LanguagePack,
    get_language_pack,
)

DEFAULT_MAX_OUTPUT_TOKENS = 1024
# Free tier dakikada 5 istekle sinirli; kisa bir bekleyisle yeniden denemek
# tum kosuyu bir kota hatasina kurban etmekten iyidir.
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 20.0


class RetrievalDecisionOut(BaseModel):
    """Gemini'den beklenen karar semasi: arama gerekli mi?"""

    needs_retrieval: bool
    search_query: str | None = None
    reason: str = Field(default="", max_length=200)


class PassageAssessmentOut(BaseModel):
    """Gemini'den beklenen karar semasi: pasajlar yeterli mi, yeni sorgu ne?"""

    sufficient: bool
    refined_query: str | None = None
    reason: str = Field(default="", max_length=200)


class SynthesisOut(BaseModel):
    answer: str


# System prompt'lar kasitli olarak Ingilizce ve dil-notr yazildi: ayni prompt
# yedi arayuz dili icin kullaniliyor ve sonuna o dilin direktifi ekleniyor.
# Onceki surum "Bir Turkce soru-cevap asistanisin" diyordu; modele Turkce
# asistan oldugunu soyleyip Fransizca cevap istemek celisikti.
#
# Arama sorgusu ile cevap dili AYRILIR. Sorgu korpusun dilinde (Turkce)
# uretilir -- BM25 tam token esleymesi yaptigi icin baska bir dilde yazilmis
# bir sorgu anahtar kelime tarafini tamamen kor birakir. Cevap ve kullaniciya
# gorunen gerekceler ise kullanicinin sectigi dilde uretilir.

_DECIDE_SYSTEM = (
    "You are the decision unit of a research agent working over a document "
    "archive. Decide whether the archive must be searched to answer the given "
    "question. Greetings, small talk, pure arithmetic and formatting requests "
    "need no search; questions asking for factual, historical, geographical or "
    "technical information do. If a search is needed, produce a short search "
    f"query reduced to keywords, written in {CORPUS_LANGUAGE_NAME}, because the "
    "archive is written in that language and keyword matching is literal. "
    "Give the reason in one short sentence."
)

_ASSESS_SYSTEM = (
    "You are the assessment unit of a research agent. Decide whether the given "
    "passages are sufficient to answer the question. If they answer it "
    "directly, set sufficient=true. If not, set sufficient=false and suggest a "
    "different search query (refined_query) that could retrieve better "
    f"results, written in {CORPUS_LANGUAGE_NAME} like the archive; leave "
    "refined_query empty when you have no new idea. Do not guess; look only at "
    "the text you were given. Give the reason in one short sentence."
)

_SYNTHESIZE_SYSTEM = (
    "You are a question answering assistant. Base your answer ONLY on the "
    "passages you were given. After each claim, append the identifier of the "
    "passage the information came from in square brackets, for example "
    "[file.md::0]. Use only the passage identifiers you were given; never "
    "invent a source and never add information that is not in the passages. "
    "Keep passage identifiers, file names and numbers exactly as they appear; "
    "do not translate or reformat them. Answer briefly and directly."
)

_DIRECT_SYSTEM = (
    "You are a helpful assistant. This question needs no document search. "
    "Answer briefly and directly. Do not cite sources unless you are making a "
    "factual claim."
)


def _with_language(system: str, pack: LanguagePack) -> str:
    """Dil direktifini system prompt'un sonuna ekler.

    Sonda duruyor cunku modeller son talimati daha guclu takip ediyor. Tek bir
    cumle oldugu icin token maliyeti ihmal edilebilir ve ek bir API cagrisi
    gerektirmez: dil destegi cagri sayisini degistirmez."""
    return system + "\n\n" + pack.directive


def _format_passages(passages: list[Passage]) -> str:
    lines = []
    for passage in passages:
        page = f", page {passage.page_number}" if passage.page_number is not None else ""
        lines.append(f"[{passage.chunk_id}] ({passage.source_file}{page}): {passage.text}")
    return "\n\n".join(lines)


class GeminiAgentLLM:
    """AgentLLM'in dort islemini Gemini uzerinden karsilar."""

    def __init__(
        self,
        client,
        model: str = "gemini-2.5-flash",
        *,
        temperature: float = 0.0,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        sleep=time.sleep,
        language: str = DEFAULT_LANGUAGE,
    ) -> None:
        self._client = client
        self.model = model
        self._pack = get_language_pack(language)
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._sleep = sleep
        self.retry_count = 0
        self.call_count = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def language(self) -> str:
        return self._pack.code

    def for_language(self, language: str) -> "GeminiAgentLLM":
        """Ayni client'i paylasan, baska dilde cevap veren bir kopya dondurur.

        Istek basina yeni bir Gemini client kurulmaz; yalnizca system prompt'a
        eklenen direktif degisir. Sayaclar paylasilmaz: her kopya kendi cagri
        sayisini tutar."""
        if get_language_pack(language).code == self._pack.code:
            return self
        return GeminiAgentLLM(
            self._client,
            self.model,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            max_retries=self._max_retries,
            retry_delay_seconds=self._retry_delay_seconds,
            sleep=self._sleep,
            language=language,
        )

    @classmethod
    def from_env(cls, settings: AgentSettings | None = None) -> "GeminiAgentLLM":
        """GEMINI_API_KEY ve GEMINI_AGENT_MODEL'i .env'den okuyarak kurar."""
        settings = settings or AgentSettings()
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY tanimli degil; Gemini agent LLM'i kurulamaz. "
                ".env dosyasina ekleyin."
            )
        from google import genai

        return cls(genai.Client(api_key=settings.gemini_api_key), model=settings.gemini_agent_model)

    def _is_retryable(self, exc: genai_errors.APIError) -> bool:
        code = getattr(exc, "code", None)
        if code is None:
            return False
        # promptevals'taki ayni mantik: Gemini'nin 4xx/5xx ayrimi
        # yeniden-denenebilirlige gore degil, bu yuzden kod dogrudan bakilir.
        return code == 429 or code >= 500

    def _call_with_retry(self, prompt: str, config):
        delay = self._retry_delay_seconds
        for attempt in range(self._max_retries + 1):
            try:
                return self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
            except genai_errors.APIError as exc:
                if not self._is_retryable(exc) or attempt == self._max_retries:
                    raise
                self.retry_count += 1
                self._sleep(delay)
                delay = min(delay * 2, 60.0)

    def _generate(self, system: str, prompt: str, schema: type[BaseModel] | None):
        self.call_count += 1
        config_kwargs = {
            "system_instruction": system,
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
            # Pydantic semasi verildiginde SDK otomatik fonksiyon cagrisi (AFC)
            # tespiti yapip her istekte uyari basiyor; burada tool cagrisi
            # kullanmadigimiz icin kapatiliyor.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = schema
        config = types.GenerateContentConfig(**config_kwargs)
        response = self._call_with_retry(prompt, config)
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self.input_tokens += getattr(usage, "prompt_token_count", 0) or 0
            self.output_tokens += getattr(usage, "candidates_token_count", 0) or 0
        return response

    def needs_retrieval(self, question: str) -> RetrievalDecision:
        response = self._generate(
            _with_language(_DECIDE_SYSTEM, self._pack),
            f"Question: {question}",
            RetrievalDecisionOut,
        )
        parsed = getattr(response, "parsed", None)
        if not isinstance(parsed, RetrievalDecisionOut):
            # Ayristirma basarisiz: aramayi denemek, cevabi uydurmaktan guvenli.
            return RetrievalDecision(
                needs_retrieval=True,
                search_query=None,
                reason=self._pack.decision_unparsed,
            )
        return RetrievalDecision(
            needs_retrieval=parsed.needs_retrieval,
            search_query=(parsed.search_query or "").strip() or None,
            reason=parsed.reason,
        )

    def assess_passages(self, question: str, passages: list[Passage]) -> PassageAssessment:
        if not passages:
            # Degerlendirilecek metin yok; API cagrisi harcamadan karar verilir.
            return PassageAssessment(
                sufficient=False, refined_query=None, reason=self._pack.no_passages_found
            )

        prompt = f"Question: {question}\n\nPassages:\n{_format_passages(passages)}"
        response = self._generate(
            _with_language(_ASSESS_SYSTEM, self._pack), prompt, PassageAssessmentOut
        )
        parsed = getattr(response, "parsed", None)
        if not isinstance(parsed, PassageAssessmentOut):
            return PassageAssessment(
                sufficient=False,
                refined_query=None,
                reason=self._pack.assessment_unparsed,
            )
        return PassageAssessment(
            sufficient=parsed.sufficient,
            refined_query=(parsed.refined_query or "").strip() or None,
            reason=parsed.reason,
        )

    def synthesize(self, question: str, passages: list[Passage]) -> str:
        prompt = (
            f"Question: {question}\n\n"
            f"Passages you may use (cite only these identifiers):\n"
            f"{_format_passages(passages)}"
        )
        response = self._generate(
            _with_language(_SYNTHESIZE_SYSTEM, self._pack), prompt, SynthesisOut
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, SynthesisOut):
            return parsed.answer
        # Sema ayristirilamadiysa duz metne dus: cevabi kaybetmek yerine
        # modelin dondurdugu metni kullan. Kaynak listesi zaten agent tarafinda
        # retrieve edilen pasajlardan uretiliyor, modelin metninden degil.
        return (getattr(response, "text", "") or "").strip()

    def answer_directly(self, question: str) -> str:
        response = self._generate(
            _with_language(_DIRECT_SYSTEM, self._pack), f"Question: {question}", None
        )
        return (getattr(response, "text", "") or "").strip()
