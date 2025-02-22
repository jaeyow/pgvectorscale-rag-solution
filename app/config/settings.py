import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(dotenv_path="./.env")


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3
    
class EmbeddingModelSettings(BaseModel):
    """Base settings for Embedding Model configurations."""
    default_model: str = Field(default="amazon.titan-embed-text-v2:0")

class OpenAISettings(LLMSettings):
    """OpenAI-specific settings extending LLMSettings."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(default="gpt-4o")
    embedding_model: str = Field(default="text-embedding-3-small")
    
class OllamaSettings(LLMSettings):
    """Ollama specific settings extending LLMSettings."""

    api_key: str = Field(default_factory=lambda: os.getenv("OLLAMA_API_KEY"))
    base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL"))
    default_model: str = Field(default="deepseek-r1:8b")
    embedding_model: str = Field(default="mxbai-embed-large:latest")
    
class BedrockSettings(LLMSettings):
    """Bedrock specific settings extending LLMSettings."""
    
    access_key: str = Field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID"))
    secret_key: str = Field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY"))
    session_token: str = Field(default_factory=lambda: os.getenv("AWS_SESSION_TOKEN"))
    region: str = Field(default_factory=lambda: os.getenv("AWS_DEFAULT_REGION"))
    default_model: str = Field(default="anthropic.claude-3-5-sonnet-20241022-v2:0")
    # embedding_model: str = Field(default="mxbai-embed-large:latest")
    max_tokens: Optional[int] = 1024

class DatabaseSettings(BaseModel):
    """Database connection settings."""

    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    table_name: str = "embeddings"
    embedding_dimensions: int = 1024 
    time_partition_interval: timedelta = timedelta(days=7)
    
class BedrockEmbeddingModelSettings(EmbeddingModelSettings):
    """Bedrock specific settings extending EmbeddingModelSettings."""
    access_key: str = Field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID"))
    secret_key: str = Field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY"))
    session_token: str = Field(default_factory=lambda: os.getenv("AWS_SESSION_TOKEN"))
    region: str = Field(default_factory=lambda: os.getenv("AWS_DEFAULT_REGION"))

class Settings(BaseModel):
    """Main settings class combining all sub-settings."""

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    llama: OllamaSettings = Field(default_factory=OllamaSettings)
    bedrock: BedrockSettings = Field(default_factory=BedrockSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    embedding_model: BedrockEmbeddingModelSettings = Field(default_factory=BedrockEmbeddingModelSettings)


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()
    return settings
