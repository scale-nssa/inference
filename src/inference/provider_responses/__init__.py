from .anthropic_message_response import AnthropicMessageResponse
from .gemini_generate_response import GeminiGenerateResponse
from .openai_completion_response import OpenAICompletionResponse

ProviderResponse = (
    OpenAICompletionResponse | AnthropicMessageResponse | GeminiGenerateResponse
)

__all__ = [
    "AnthropicMessageResponse",
    "GeminiGenerateResponse",
    "OpenAICompletionResponse",
    "ProviderResponse",
]
