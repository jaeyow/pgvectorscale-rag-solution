from openai import OpenAI
from services.TitanEmbeddings import TitanEmbeddings

####################
# This is where we register the embedding mdoel clients that can be used by the EmbeddingModelFactory.
# The EmbeddingModelFactory will use the registered clients to create the appropriate embedding mdoel client.
####################

EMBEDDING_MODELS = {}


def register_embedding_model_client(embedding_model_client):
    """
    Register an embedding model client.
    """

    def decorator(fn):
        """
        Decorator function to register an embedding model client.
        """
        EMBEDDING_MODELS[embedding_model_client] = fn
        return fn

    return decorator


def get_embedding_model_client(embedding_model_client):
    """
    Retrieve an embedding model client by name.
    """
    try:
        return EMBEDDING_MODELS[embedding_model_client]
    except KeyError:
        raise ValueError(f"LLM client '{embedding_model_client}' is not registered.")


# Register embedding model clients
# Add your embedding model client registration functions here and they will be automatically registered
# as this module is imported. There is no need to touch the EmbeddingModelFactory class.


@register_embedding_model_client("openai")
def openai_client(settings):
    """
    Create an OpenAI embedding model client.
    """
    return OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,  # required, but unused
    )

@register_embedding_model_client("llama")
def llama_client(settings):
    """
    Create an Ollama embedding model client.
    Note that Ollama is OpenAI compatible, so we can continue use the OpenAI client.
    """
    return OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,  # required, but unused
    )
    
@register_embedding_model_client("bedrock_embedding_model")
def bedrock_client(settings):
    """
    Create a Bedrock embedding model client
    """
    bedrock_params = {
        "aws_access_key_id": settings.access_key,
        "aws_secret_access_key": settings.secret_key,
        "aws_session_token": settings.session_token,
        "region_name": settings.region,
    }
    
    print(f"Settings: {settings}")
    print(f"Default embedding model: {settings.default_model}")
    
    return TitanEmbeddings(model_id=settings.default_model, **bedrock_params)
