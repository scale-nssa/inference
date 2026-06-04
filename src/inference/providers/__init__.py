from .base import Model
from .claude_vertex import ClaudeVertexModel
from .gemini_vertex import GeminiVertexModel
from .lm_studio import LMStudioModel
from .openrouter import OpenRouterModel

PROVIDERS: dict[str, type[Model]] = {
    "claude_vertex": ClaudeVertexModel,
    "gemini_vertex": GeminiVertexModel,
    "lm_studio": LMStudioModel,
    "openrouter": OpenRouterModel,
}

__all__ = [
    "ClaudeVertexModel",
    "GeminiVertexModel",
    "LMStudioModel",
    "Model",
    "OpenRouterModel",
    "PROVIDERS",
]
