"""Agent katmaninin ayarlari.

RAG'in kendi Settings'inden ayri tutulur: agent yalnizca Gemini free tier ile
calisiyor ve zorunlu bir Anthropic key'e ihtiyac duymamali."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str | None = None
    # Google AI Studio free tier'da ucretsiz olan model; promptevals projesinin
    # GeminiProviderConfig varsayilaniyla ayni.
    gemini_agent_model: str = "gemini-2.5-flash"
