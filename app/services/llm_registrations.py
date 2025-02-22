import anthropic
import instructor
from openai import OpenAI

####################
# This is where we register the LLM clients that can be used by the LLMFactory.
# The LLMFactory will use the registered clients to create the appropriate LLM client.
####################

LLMS = {}


def register_llm_client(llm_client):
    """
    Register a function to create an LLM client.
    """

    def decorator(fn):
        """
        Decorator function to register an LLM client.
        """
        LLMS[llm_client] = fn
        return fn

    return decorator


def get_llm_client(llm_client):
    """
    Retrieve an LLM client by name.
    """
    try:
        return LLMS[llm_client]
    except KeyError:
        raise ValueError(f"LLM client '{llm_client}' is not registered.")


# Register LLM clients
# Add your LLM client registration functions here and they will be automatically registered
# as this module is imported. There is no need to touch the LLMFactory class.


@register_llm_client("openai")
def openai_client(settings):
    """
    Create an OpenAI LLM client.
    """
    return instructor.from_openai(OpenAI(api_key=settings.api_key))


@register_llm_client("anthropic")
def anthropic_client(settings):
    """
    Create an Anthropic LLM client.
    """
    return instructor.from_anthropic(anthropic.Anthropic(api_key=settings.api_key))


@register_llm_client("llama")
def llama_client(settings):
    """
    Create an Ollama LLM client.
    Note that Ollama is OpenAI compatible, so we can continue use the OpenAI client.
    """
    return instructor.from_openai(
        OpenAI(base_url=settings.base_url, api_key=settings.api_key),
        mode=instructor.Mode.JSON,
    )


@register_llm_client("bedrock")
def bedrock_client(settings):
    """
    Create a Bedrock LLM client (for Anthropic Claude models)
    """
    client = anthropic.AnthropicBedrock(
        aws_access_key=settings.access_key,
        aws_secret_key=settings.secret_key,
        aws_session_token=settings.session_token,
        aws_region=settings.region,
    )

    instructor_client = instructor.from_anthropic(client)
    return instructor_client
