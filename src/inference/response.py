from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .logprobs import TokenLogprob
from .prompt import Prompt
from .provider_responses import ProviderResponse
from .token_stats import TokenStats


class Response(BaseModel):
    """Lean response from `model.ask()`. Coerces to its text via `str()`."""

    model_config = ConfigDict(frozen=True)

    text: str
    thinking: str | None = None
    input_tokens: int
    output_tokens: int
    thinking_tokens: int | None = None
    logprobs: tuple[TokenLogprob, ...] | None = None

    def __str__(self) -> str:
        return self.text


class LLMOutput(BaseModel):
    """Full response from `llm.call()`. Includes prompt + typed provider payload."""

    model_config = ConfigDict(frozen=True)

    prompt: Prompt
    text: str
    thinking: str | None
    tokens: TokenStats
    provider_response: ProviderResponse
    logprobs: tuple[TokenLogprob, ...] | None = None

    def __str__(self) -> str:
        return self.text

    def to_response(self) -> Response:
        return Response(
            text=self.text,
            thinking=self.thinking,
            input_tokens=self.tokens.input_tokens,
            output_tokens=self.tokens.output_tokens,
            thinking_tokens=self.tokens.thinking_tokens,
            logprobs=self.logprobs,
        )
