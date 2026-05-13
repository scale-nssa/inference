from typing import Any

from pydantic import BaseModel, ConfigDict


class OpenAIMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    role: str
    content: str | None = None
    refusal: str | None = None
    reasoning: str | None = None


class OpenAIChoice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    index: int
    finish_reason: str
    message: OpenAIMessage
    logprobs: dict[str, Any] | None = None


class OpenAIUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: dict[str, Any] | None = None
    prompt_tokens_details: dict[str, Any] | None = None


class OpenAICompletionResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    object: str
    created: int
    model: str
    choices: tuple[OpenAIChoice, ...]
    usage: OpenAIUsage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None
