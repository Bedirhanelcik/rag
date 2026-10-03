from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Yapilandirmada beklenen embedding modeli. `retrieval.embeddings` ayni degeri
#: kendi varsayilani olarak tasir; buradaki yalnizca hata mesaji icin.
_EXPECTED_EMBEDDING_MODEL = "gemini-embedding-001"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LEGACY -- yalnizca eski /query ucu icin. Production path (web -> /agent/ask)
    # Anthropic'i hic kullanmaz; `anthropic` paketi de artik uretim
    # bagimliligi degil (opsiyonel `legacy` extra). Her ikisinin de varsayilani
    # oldugu icin dagitimda tanimlanmasi gerekmez.
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    # Embedding Gemini API uzerinden uretilir: yerel bir ML modeli
    # yuklenmedigi icin surec bellegi 512 MB sinirinin altinda kalir.
    gemini_api_key: str | None = None
    embedding_model_name: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    chroma_persist_dir: str = "data/chroma"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k_vector: int = 10
    top_k_keyword: int = 10
    top_k_final: int = 5
    rrf_k: int = 60
    # Upload varsayilan olarak KAPALI: mevcut Chroma koleksiyonu tum ziyaretciler
    # arasinda paylasimli, yani kullanici izolasyonu yok. Yerel gelistirmede
    # UPLOAD_ENABLED=true ile acilir; dagitimda kapali kalmali.
    upload_enabled: bool = False
    # /ingest operator ucudur ve varsayilan olarak KAPALIDIR: token tanimli
    # degilse uc nokta hizmet vermez. Boylece yapilandirilmamis bir dagitimda
    # korpusa herkes yazamaz.
    ingest_api_token: str | None = None
    upload_max_bytes: int = 10 * 1024 * 1024
    # Virgulle ayrilmis frontend origin listesi. Bos birakilirsa CORS
    # middleware'i hic eklenmez: onerilen dagitimda tarayici backend'e dogrudan
    # konusmaz, istekler Next.js sunucu tarafindan proxy'lenir.
    allowed_origins: str = ""

    @field_validator("embedding_model_name")
    @classmethod
    def _validate_embedding_model(cls, value: str) -> str:
        """Yerel (HuggingFace) model adini acilista reddet.

        Onceki surum embedding'i yerel bir `sentence-transformers` modeliyle
        uretiyordu ve `EMBEDDING_MODEL_NAME` bir HuggingFace adiydi. O deger
        ortamda kalirsa uygulama sorunsuz aciliyor, `/health` "ok" donuyor ve
        hata ancak ilk embedding cagrisinda -- yani ilk kullanici sorusunda --
        ortaya cikiyor. Yapilandirma hatasi acilista gorulmeli: bozuk bir
        dagitim trafigi hic almasin."""
        name = value.strip()
        if not name:
            raise ValueError(
                f"EMBEDDING_MODEL_NAME bos olamaz; ornek: {_EXPECTED_EMBEDDING_MODEL}"
            )
        if "/" in name:
            raise ValueError(
                f"EMBEDDING_MODEL_NAME bir Gemini embedding modeli olmali, "
                f"HuggingFace model adi degil (verilen: {name!r}). "
                f"Ornek: {_EXPECTED_EMBEDDING_MODEL}"
            )
        return name

    @property
    def allowed_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]
