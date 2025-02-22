from typing import List
from config.settings import get_settings
import services.embedding_model_registrations as embedding_model_registrations

class EmbeddingModelFactory:
    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        print(f"Settings for provider {self.provider}: {self.settings}")
        self.client = embedding_model_registrations.get_embedding_model_client(self.provider)(self.settings)

    def create_embedding(self, text: str, **kwargs) -> List[float]:
        
        if self.provider == "bedrock_embedding_model":
            return self.client(text)
        else:
            embedding_params = {
                "model": kwargs.get("model", self.settings.embedding_model),
                "input": [text],
            }
            print(f"Embedding params: {embedding_params}")
            
            return (
                    self.client.embeddings.create(**embedding_params)
                    .data[0]
                    .embedding
                )
