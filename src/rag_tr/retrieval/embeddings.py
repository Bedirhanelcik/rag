"""Embedding uretimi -- Gemini embedding API uzerinden.

Neden yerel model degil: `intfloat/multilingual-e5-small` 118M parametredir
(250k'lik sozluk yuzunden embedding matrisi tek basina ~384 MB fp32) ve
`sentence-transformers` ile yuklendiginde surec RSS'i olculmus degerlerle
~1.26 GB'a cikiyor (torch +154 MB, transformers +225 MB, model +789 MB). 512 MB
bellekli bir ortamda bu mumkun degil. Ayni modeli ONNX'e cevirmek de yetmez:
fp32 agirliklar tek basina butceyi asiyor.

Gemini embedding API'si bu maliyeti tamamen kaldirir: surecte ML calisma zamani
yok, taban kullanim ~88 MB'da kaliyor. Hibrit retrieval mimarisi (Chroma + BM25
+ RRF) aynen korunur; degisen tek sey vektorlerin nerede uretildigi.

Iki onemli davranis korunuyor:

  * **Asimetrik kodlama.** e5 modelinde bu `"query: "` / `"passage: "` onekiyle
    yapiliyordu; Gemini'de dogru karsiligi `task_type` alanidir
    (RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT). Sorgu ve pasaj farkli kodlanmaya
    devam eder.
  * **L2 normalizasyon.** Chroma koleksiyonu varsayilan L2 uzakligi kullaniyor;
    birim vektorlerde L2 siralamasi cosine siralamasiyla ayni oldugu icin
    mevcut siralama anlamini korumak adina vektorler normalize edilir.

Istemci tembel kurulur: uygulama acilisi (ve `/health`) ne ag erisimi ne de API
anahtari gerektirir.
"""

import numpy as np

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIMENSIONS = 768

#: Tek istekte gonderilen en fazla metin sayisi. Buyuk ingest'ler bu boyda
#: parcalara bolunur.
DEFAULT_BATCH_SIZE = 32

_QUERY_TASK = "RETRIEVAL_QUERY"
_DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"


class EmbeddingModel:
    """Arayuzu onceki yerel model implementasyonuyla birebir aynidir:
    `encode_passages(list[str]) -> np.ndarray` ve `encode_query(str) ->
    np.ndarray`. Boylece `RAGService` ve testler degismeden calisir."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        api_key: str | None = None,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        client=None,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self._api_key = api_key
        self._batch_size = max(1, batch_size)
        self._client = client

    def _ensure_client(self):
        """Istemciyi ilk kullanimda kurar; acilista ag/anahtar gerekmez."""
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY tanımlı değil; embedding üretilemez. "
                "Sunucu tarafı ortam değişkenlerine ekleyin."
            )
        from google import genai

        self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _check_model_name(self) -> None:
        """Yerel model adi birakilmissa net bir hata ver.

        Onceki surumde `EMBEDDING_MODEL_NAME` bir HuggingFace model adiydi
        (orn. `intfloat/multilingual-e5-small`). Bu deger Gemini'ye model adi
        olarak gonderilirse anlamsiz bir API hatasi doner; bunun yerine ne
        yapilmasi gerektigini soyluyoruz."""
        if "/" in self.model_name:
            raise RuntimeError(
                f"EMBEDDING_MODEL_NAME bir Gemini embedding modeli olmali, "
                f"HuggingFace model adi degil (verilen: {self.model_name!r}). "
                f"Ornek: {DEFAULT_EMBEDDING_MODEL}"
            )

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        self._check_model_name()

        from google.genai import types

        client = self._ensure_client()
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.dimensions,
        )

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = client.models.embed_content(
                model=self.model_name, contents=batch, config=config
            )
            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Embedding yanıtı beklenen sayıda vektör taşımıyor: "
                    f"{len(embeddings)} != {len(batch)}"
                )
            vectors.extend(list(embedding.values) for embedding in embeddings)

        matrix = np.asarray(vectors, dtype=np.float32)
        return _l2_normalize(matrix)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(list(texts), _DOCUMENT_TASK)

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed([text], _QUERY_TASK)[0]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Satir bazinda birim uzunluga getirir. Sifir vektorler oldugu gibi kalir."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
