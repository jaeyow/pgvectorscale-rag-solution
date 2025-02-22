import os

from pydantic import BaseModel, Field


class EmbeddingModelSettings(BaseModel):
    """Base settings for Embedding Model configurations."""

    default_model: str = Field(default="text-embedding-3-small")


class OpenAIEmbeddingModelSettings(EmbeddingModelSettings):
    """OpenAI-specific settings extending EmbeddingModelSettings."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(default="text-embedding-3-small")


class OllamaEmbeddingModelSettings(EmbeddingModelSettings):
    """Ollama specific settings extending EmbeddingModelSettings."""

    api_key: str = Field(default_factory=lambda: os.getenv("OLLAMA_API_KEY"))
    base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL"))
    default_model: str = Field(default="mxbai-embed-large:latest")


class BedrockEmbeddingModelSettings(EmbeddingModelSettings):
    """Bedrock specific settings extending EmbeddingModelSettings."""

    default_model: str = Field(default="amazon.titan-embed-text-v2:0")
    access_key: str = Field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID"))
    secret_key: str = Field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY"))
    session_token: str = Field(default_factory=lambda: os.getenv("AWS_SESSION_TOKEN"))
    region: str = Field(default_factory=lambda: os.getenv("AWS_DEFAULT_REGION"))
