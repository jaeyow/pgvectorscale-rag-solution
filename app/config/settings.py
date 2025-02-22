import logging
import os
from datetime import timedelta
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from config.embedding_model_settings import (
    BedrockEmbeddingModelSettings,
    OllamaEmbeddingModelSettings,
    OpenAIEmbeddingModelSettings,
)
from config.llm_settings import BedrockSettings, OllamaSettings, OpenAISettings

load_dotenv(dotenv_path="./.env")


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class DatabaseSettings(BaseModel):
    """Database connection settings."""

    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    table_name: str = "embeddings"
    embedding_dimensions: int = 1024
    time_partition_interval: timedelta = timedelta(days=7)


class Settings(BaseModel):
    """Main settings class combining all sub-settings."""

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    llama: OllamaSettings = Field(default_factory=OllamaSettings)
    bedrock: BedrockSettings = Field(default_factory=BedrockSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    openai_embedding_model: OpenAIEmbeddingModelSettings = Field(
        default_factory=OpenAIEmbeddingModelSettings
    )
    llama_embedding_model: OllamaEmbeddingModelSettings = Field(
        default_factory=OllamaEmbeddingModelSettings
    )
    bedrock_embedding_model: BedrockEmbeddingModelSettings = Field(
        default_factory=BedrockEmbeddingModelSettings
    )


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()
    return settings
