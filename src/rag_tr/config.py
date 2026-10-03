from pydantic_settings import BaseSettings, SettingsConfigDict


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

    @property
    def allowed_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]
