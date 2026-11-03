from .base import Model
from .claude_vertex import ClaudeVertexModel
from .gemini_vertex import GeminiVertexModel
from .gpt_oss_vertex import GptOssVertexModel
from .stanford import StanfordModel

PROVIDERS: dict[str, type[Model]] = {
    "claude_vertex": ClaudeVertexModel,
    "gemini_vertex": GeminiVertexModel,
    "gpt_oss_vertex": GptOssVertexModel,
    "stanford": StanfordModel,
}

__all__ = [
    "ClaudeVertexModel",
    "GeminiVertexModel",
    "GptOssVertexModel",
    "Model",
    "PROVIDERS",
    "StanfordModel",
]
