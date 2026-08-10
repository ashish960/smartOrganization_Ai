from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    APP_NAME: str = "SmartOrg AI Service"
    PORT: int = 8000
    DEBUG: bool = False

    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"

    PINECONE_API_KEY: str
    PINECONE_INDEX: str = "smartorg-docs"

    INTERNAL_API_KEY: str = "smartorg-internal-secret-2024"

    NODE_BACKEND_URL: str = "http://localhost:5000"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
