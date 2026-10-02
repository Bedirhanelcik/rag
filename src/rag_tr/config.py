from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Bos varsayilan: agent-only dagitimda (yalnizca Gemini) anlamsiz bir dummy
    # sir tutmak gerekmesin. Claude yolu (/query) bu key ile cagri yapar;
    # bos birakilirsa yalnizca o cagri sirasinda kimlik hatasi alinir.
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    embedding_model_name: str = "intfloat/multilingual-e5-small"
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
    upload_max_bytes: int = 10 * 1024 * 1024
    # Virgulle ayrilmis frontend origin listesi. Bos birakilirsa CORS
    # middleware'i hic eklenmez: onerilen dagitimda tarayici backend'e dogrudan
    # konusmaz, istekler Next.js sunucu tarafindan proxy'lenir.
    allowed_origins: str = ""

    @property
    def allowed_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]
