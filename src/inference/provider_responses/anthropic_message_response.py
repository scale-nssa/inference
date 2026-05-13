from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class AnthropicUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    service_tier: str | None = None


class AnthropicContentBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    type: str
    text: str | None = None
    thinking: str | None = None
    signature: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_signature(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        signature = value.get("signature")
        if isinstance(signature, bytes):
            return {**value, "signature": signature.decode("utf-8", errors="ignore")}
        return value


class AnthropicMessageResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    type: str
    role: str
    model: str
    content: tuple[AnthropicContentBlock, ...]
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage
