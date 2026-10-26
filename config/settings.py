from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Required external-provider credentials for the LLM integrations."""

    GROQ_API_KEY: str
    GEMINI_API_KEY: str

    model_config = SettingsConfigDict(
        env_file= BASE_DIR / '.env',
        env_file_encoding= 'utf-8',
    )


# Resolve settings once so missing credentials fail clearly during startup.
settings = Settings()