"""Cevap dili.

Arayuz yedi dil sunuyor ve kullanicinin sectigi dil `/agent/ask` istegiyle
birlikte geliyor. Bu modul o secimin iki sonucunu tasir:

1. Modele verilen dil direktifi. Cevap, karar gerekcesi ve degerlendirme
   gerekcesi kullanicinin dilinde uretilir.
2. Kod tarafindan uretilen, kullaniciya gorunen sabit metinler. Adim
   detaylari ("3 pasaj bulundu"), baglam bulunamadi mesaji ve tool hatasi
   mesaji modelden degil buradan gelir; dolayisiyla onlarin da cevirisi
   burada durur.

ONEMLI -- arama sorgusu cevrilmez. Korpus Turkce ve BM25 tam token esleymesi
yapiyor: Fransizca bir sorguyla Turkce bir dokumani anahtar kelimeyle bulmak
mumkun degil. Bu yuzden modele arama sorgusunu KORPUS dilinde (Turkce)
yazmasi, cevabi ise KULLANICININ dilinde vermesi soylenir. Bu ayrim sayesinde
hibrit retrieval bozulmaz ve ek bir ceviri API cagrisi da gerekmez: dil
yalnizca mevcut cagrilarin system prompt'una giren bir cumledir.
"""

from dataclasses import dataclass

#: Arayuzun sundugu diller. Sira arayuzdeki secici sirasiyla ayni.
SUPPORTED_LANGUAGES: tuple[str, ...] = ("tr", "en", "fr", "ar", "es", "zh", "hi")

#: Istek dil tasimadiginda kullanilan dil. Korpusun dili de bu.
DEFAULT_LANGUAGE = "tr"

#: Dokumanlarin dili. Arama sorgusu bu dilde uretilir.
CORPUS_LANGUAGE_NAME = "Turkish"


@dataclass(frozen=True)
class LanguagePack:
    """Tek bir dil icin model direktifi ve sabit kullanici metinleri."""

    code: str
    #: Dilin kendi adindaki hali; loglar ve prompt icin.
    name: str
    #: Modele verilen, "kullaniciya su dilde cevap ver" direktifi.
    directive: str
    #: Yeterli baglam bulunamadiginda donen cevap.
    no_context: str
    #: Retrieval katmani hata verdiginde donen cevap.
    tool_failure: str
    #: Model gerekce dondurmediginde kullanilan yedek metinler.
    search_needed: str
    search_not_needed: str
    answered_without_retrieval: str
    query_refined: str
    #: Sablonlar. Yalnizca asagidaki alan adlari kullanilabilir.
    searched_attempt: str  # {attempt}, {count}
    answered_from_context: str  # {count}
    insufficient_after_searches: str  # {searches}
    tool_error: str  # {error}
    #: Model yanitinin semaya uymadigi, nadir durumlar.
    decision_unparsed: str
    assessment_unparsed: str
    no_passages_found: str


_PACKS: dict[str, LanguagePack] = {
    "tr": LanguagePack(
        code="tr",
        name="Türkçe",
        directive="Kullanıcıya Türkçe cevap ver.",
        no_context="Dokümanlarda bu bilgi yok.",
        tool_failure="Belge araması sırasında bir hata oluştu, bu nedenle cevap üretilemedi.",
        search_needed="arama gerekli",
        search_not_needed="arama gerekmiyor",
        answered_without_retrieval="doküman araması yapılmadan cevaplandı",
        query_refined="sorgu yeniden formüllendi",
        searched_attempt="deneme {attempt}: {count} pasaj bulundu",
        answered_from_context="{count} pasaj temel alınarak cevaplandı",
        insufficient_after_searches="{searches} arama sonrası yeterli bağlam bulunamadı",
        tool_error="retrieval tool hatası: {error}",
        decision_unparsed="karar ayrıştırılamadı, arama yapılıyor",
        assessment_unparsed="değerlendirme ayrıştırılamadı",
        no_passages_found="pasaj bulunamadı",
    ),
    "en": LanguagePack(
        code="en",
        name="English",
        directive="Answer the user in English.",
        no_context="That information is not in the documents.",
        tool_failure="The document search failed, so no answer could be produced.",
        search_needed="search required",
        search_not_needed="no search required",
        answered_without_retrieval="answered without searching the documents",
        query_refined="query reformulated",
        searched_attempt="attempt {attempt}: {count} passages found",
        answered_from_context="answered from {count} passages",
        insufficient_after_searches="no sufficient context after {searches} searches",
        tool_error="retrieval tool error: {error}",
        decision_unparsed="decision could not be parsed, searching anyway",
        assessment_unparsed="assessment could not be parsed",
        no_passages_found="no passages found",
    ),
    "fr": LanguagePack(
        code="fr",
        name="Français",
        directive="Répondez à l'utilisateur en français.",
        no_context="Cette information ne figure pas dans les documents.",
        tool_failure="La recherche documentaire a échoué, aucune réponse n'a pu être produite.",
        search_needed="recherche nécessaire",
        search_not_needed="recherche non nécessaire",
        answered_without_retrieval="réponse donnée sans consulter les documents",
        query_refined="requête reformulée",
        searched_attempt="tentative {attempt} : {count} passages trouvés",
        answered_from_context="réponse fondée sur {count} passages",
        insufficient_after_searches="contexte insuffisant après {searches} recherches",
        tool_error="erreur de l'outil de retrieval : {error}",
        decision_unparsed="décision illisible, recherche lancée malgré tout",
        assessment_unparsed="évaluation illisible",
        no_passages_found="aucun passage trouvé",
    ),
    "ar": LanguagePack(
        code="ar",
        name="العربية",
        directive="أجب المستخدم باللغة العربية.",
        no_context="هذه المعلومة غير موجودة في المستندات.",
        tool_failure="فشل البحث في المستندات، لذلك لم يتمكن النظام من إنتاج إجابة.",
        search_needed="البحث مطلوب",
        search_not_needed="البحث غير مطلوب",
        answered_without_retrieval="تمت الإجابة دون البحث في المستندات",
        query_refined="تمت إعادة صياغة الاستعلام",
        searched_attempt="المحاولة {attempt}: تم العثور على {count} مقطعًا",
        answered_from_context="تمت الإجابة استنادًا إلى {count} مقطعًا",
        insufficient_after_searches="لا يوجد سياق كافٍ بعد {searches} عمليات بحث",
        tool_error="خطأ في أداة الـ retrieval: {error}",
        decision_unparsed="لم يتم تحليل القرار، وسيتم البحث على أي حال",
        assessment_unparsed="لم يتم تحليل التقييم",
        no_passages_found="لم يتم العثور على مقاطع",
    ),
    "es": LanguagePack(
        code="es",
        name="Español",
        directive="Responde al usuario en español.",
        no_context="Esa información no está en los documentos.",
        tool_failure="La búsqueda en los documentos falló, por lo que no se pudo generar una respuesta.",
        search_needed="se requiere búsqueda",
        search_not_needed="no se requiere búsqueda",
        answered_without_retrieval="respondido sin buscar en los documentos",
        query_refined="consulta reformulada",
        searched_attempt="intento {attempt}: {count} pasajes encontrados",
        answered_from_context="respondido a partir de {count} pasajes",
        insufficient_after_searches="contexto insuficiente tras {searches} búsquedas",
        tool_error="error de la herramienta de retrieval: {error}",
        decision_unparsed="no se pudo interpretar la decisión, se busca de todos modos",
        assessment_unparsed="no se pudo interpretar la evaluación",
        no_passages_found="no se encontraron pasajes",
    ),
    "zh": LanguagePack(
        code="zh",
        name="中文（简体）",
        directive="请用简体中文回答用户。",
        no_context="文档中没有这项信息。",
        tool_failure="文档检索失败，因此无法生成答案。",
        search_needed="需要检索",
        search_not_needed="无需检索",
        answered_without_retrieval="未检索文档即作答",
        query_refined="已重写查询",
        searched_attempt="第 {attempt} 次尝试：找到 {count} 个片段",
        answered_from_context="依据 {count} 个片段作答",
        insufficient_after_searches="经过 {searches} 次检索仍无足够上下文",
        tool_error="retrieval 工具错误：{error}",
        decision_unparsed="无法解析判断结果，仍执行检索",
        assessment_unparsed="无法解析评估结果",
        no_passages_found="未找到片段",
    ),
    "hi": LanguagePack(
        code="hi",
        name="हिन्दी",
        directive="उपयोगकर्ता को हिन्दी में उत्तर दें।",
        no_context="यह जानकारी दस्तावेज़ों में नहीं है।",
        tool_failure="दस्तावेज़ खोज विफल रही, इसलिए उत्तर तैयार नहीं किया जा सका।",
        search_needed="खोज आवश्यक है",
        search_not_needed="खोज आवश्यक नहीं है",
        answered_without_retrieval="दस्तावेज़ों में खोज किए बिना उत्तर दिया गया",
        query_refined="प्रश्न फिर से तैयार किया गया",
        searched_attempt="प्रयास {attempt}: {count} अंश मिले",
        answered_from_context="{count} अंशों के आधार पर उत्तर दिया गया",
        insufficient_after_searches="{searches} खोजों के बाद भी पर्याप्त संदर्भ नहीं मिला",
        tool_error="retrieval टूल त्रुटि: {error}",
        decision_unparsed="निर्णय पढ़ा नहीं जा सका, फिर भी खोज की जा रही है",
        assessment_unparsed="मूल्यांकन पढ़ा नहीं जा सका",
        no_passages_found="कोई अंश नहीं मिला",
    ),
}


def normalize_language(code: str | None) -> str:
    """Desteklenen bir dil koduna indirger.

    Bilinmeyen veya bos bir deger sessizce varsayilana duser: dil secimi
    gorsel bir tercih, istegi reddetmek icin bir sebep degil. HTTP katmani
    yine de semada dogrulama yapar, bu yalnizca ikinci savunma katmani.
    `en-US` gibi bolgeli kodlarin yalnizca dil kismi dikkate alinir."""
    if not code:
        return DEFAULT_LANGUAGE
    base = code.strip().replace("_", "-").split("-")[0].lower()
    return base if base in _PACKS else DEFAULT_LANGUAGE


def get_language_pack(code: str | None) -> LanguagePack:
    return _PACKS[normalize_language(code)]
