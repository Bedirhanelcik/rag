"""AgentLLM'in Gemini (Google AI Studio free tier) implementasyonu.

ResearchAgent bu modulu tanimaz: yalnizca dort metotluk AgentLLM yuzeyini
cagirir. Saglayici degistirmek icin dongude tek satir degismesi gerekmez.

Kararlar structured output (response_schema) ile alinir, serbest metin
ayristirilmaz; sicaklik 0 ve ayristirma hatalarinda guvenli tarafa dusen
varsayilanlar sayesinde davranis olabildigince deterministiktir.
"""

from google.genai import types
from pydantic import BaseModel, Field

from rag_tr.agent.config import AgentSettings
from rag_tr.agent.contracts import PassageAssessment, RetrievalDecision
from rag_tr.contracts import Passage

DEFAULT_MAX_OUTPUT_TOKENS = 1024


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


_DECIDE_SYSTEM = (
    "Bir Türkçe doküman arşivi üzerinde çalışan araştırma ajanının karar "
    "birimisin. Verilen soruyu cevaplamak için arşivde arama yapılması gerekip "
    "gerekmediğine karar ver. Selamlama, sohbet, saf matematik veya biçim "
    "değiştirme istekleri için arama gerekmez; olgusal, tarihsel, coğrafi veya "
    "teknik bilgi isteyen sorular için gerekir. Arama gerekiyorsa, soruyu "
    "anahtar kelimelere indirgeyen kısa bir arama sorgusu üret. Gerekçeyi tek "
    "kısa cümleyle yaz."
)

_ASSESS_SYSTEM = (
    "Bir araştırma ajanının değerlendirme birimisin. Verilen pasajların soruyu "
    "cevaplamaya yeterli olup olmadığına karar ver. Pasajlar soruyu doğrudan "
    "cevaplıyorsa sufficient=true. Yetersizse sufficient=false ver ve daha iyi "
    "sonuç getirebileceğini düşündüğün farklı bir arama sorgusu öner "
    "(refined_query); yeni bir fikrin yoksa refined_query'yi boş bırak. "
    "Tahminde bulunma, yalnızca verilen metne bak. Gerekçeyi tek kısa cümleyle yaz."
)

_SYNTHESIZE_SYSTEM = (
    "Bir Türkçe soru-cevap asistanısın. Cevabını YALNIZCA sana verilen "
    "pasajlardaki bilgiye dayandır. Her iddianın sonuna, bilgiyi aldığın "
    "pasajın kimliğini köşeli parantez içinde ekle, örneğin [dosya.md::0]. "
    "Yalnızca sana verilen pasaj kimliklerini kullan; yeni kaynak uydurma ve "
    "pasajlarda bulunmayan bilgi ekleme. Kısa ve doğrudan cevap ver."
)

_DIRECT_SYSTEM = (
    "Türkçe konuşan yardımcı bir asistansın. Bu soru doküman araması "
    "gerektirmiyor. Kısa ve doğrudan cevap ver. Olgusal bir iddiada "
    "bulunmuyorsan kaynak gösterme."
)


def _format_passages(passages: list[Passage]) -> str:
    lines = []
    for passage in passages:
        page = f", sayfa {passage.page_number}" if passage.page_number is not None else ""
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
    ) -> None:
        self._client = client
        self.model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self.call_count = 0

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
        return self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

    def needs_retrieval(self, question: str) -> RetrievalDecision:
        response = self._generate(_DECIDE_SYSTEM, f"Soru: {question}", RetrievalDecisionOut)
        parsed = getattr(response, "parsed", None)
        if not isinstance(parsed, RetrievalDecisionOut):
            # Ayristirma basarisiz: aramayi denemek, cevabi uydurmaktan guvenli.
            return RetrievalDecision(
                needs_retrieval=True,
                search_query=None,
                reason="karar ayrıştırılamadı, arama yapılıyor",
            )
        return RetrievalDecision(
            needs_retrieval=parsed.needs_retrieval,
            search_query=(parsed.search_query or "").strip() or None,
            reason=parsed.reason,
        )

    def assess_passages(self, question: str, passages: list[Passage]) -> PassageAssessment:
        if not passages:
            # Degerlendirilecek metin yok; API cagrisi harcamadan karar verilir.
            return PassageAssessment(sufficient=False, refined_query=None, reason="pasaj bulunamadı")

        prompt = f"Soru: {question}\n\nPasajlar:\n{_format_passages(passages)}"
        response = self._generate(_ASSESS_SYSTEM, prompt, PassageAssessmentOut)
        parsed = getattr(response, "parsed", None)
        if not isinstance(parsed, PassageAssessmentOut):
            return PassageAssessment(
                sufficient=False,
                refined_query=None,
                reason="değerlendirme ayrıştırılamadı",
            )
        return PassageAssessment(
            sufficient=parsed.sufficient,
            refined_query=(parsed.refined_query or "").strip() or None,
            reason=parsed.reason,
        )

    def synthesize(self, question: str, passages: list[Passage]) -> str:
        prompt = (
            f"Soru: {question}\n\n"
            f"Kullanabileceğin pasajlar (yalnızca bu kimlikleri kaynak göster):\n"
            f"{_format_passages(passages)}"
        )
        response = self._generate(_SYNTHESIZE_SYSTEM, prompt, SynthesisOut)
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, SynthesisOut):
            return parsed.answer
        # Sema ayristirilamadiysa duz metne dus: cevabi kaybetmek yerine
        # modelin dondurdugu metni kullan. Kaynak listesi zaten agent tarafinda
        # retrieve edilen pasajlardan uretiliyor, modelin metninden degil.
        return (getattr(response, "text", "") or "").strip()

    def answer_directly(self, question: str) -> str:
        response = self._generate(_DIRECT_SYSTEM, f"Soru: {question}", None)
        return (getattr(response, "text", "") or "").strip()
