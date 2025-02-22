from typing import List
from config.settings import get_settings
import services.embedding_model_registrations as embedding_model_registrations


class EmbeddingModelFactory:
    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = embedding_model_registrations.get_embedding_model_client(
            self.provider
        )(self.settings)

    def create_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate an embedding for the given text.
        """
        embedding_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "input": [text],
        }

        return self.client.embeddings.create(**embedding_params).data[0].embedding
