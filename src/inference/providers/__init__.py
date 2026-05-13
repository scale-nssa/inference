from .base import Model
from .claude_vertex import ClaudeVertexModel
from .gemini_vertex import GeminiVertexModel
from .lm_studio import LMStudioModel

PROVIDERS: dict[str, type[Model]] = {
    "claude_vertex": ClaudeVertexModel,
    "gemini_vertex": GeminiVertexModel,
    "lm_studio": LMStudioModel,
}

__all__ = [
    "ClaudeVertexModel",
    "GeminiVertexModel",
    "LMStudioModel",
    "Model",
    "PROVIDERS",
]
