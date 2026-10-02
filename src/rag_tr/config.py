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
