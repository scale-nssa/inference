from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class GeminiPart(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    text: str | None = None
    thought: bool | None = None
    thought_signature: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_thought_signature(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        ts = value.get("thought_signature")
        if isinstance(ts, bytes):
            return {**value, "thought_signature": ts.decode("utf-8", errors="ignore")}
        return value


class GeminiContent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    parts: tuple[GeminiPart, ...] | None = None
    role: str | None = None


class GeminiCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    content: GeminiContent | None = None
    finish_reason: str | None = None
    finish_message: str | None = None
    index: int | None = None
    safety_ratings: tuple[dict[str, Any], ...] | None = None
    avg_logprobs: float | None = None
    token_count: int | None = None


class GeminiUsageMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    prompt_token_count: int | None = None
    candidates_token_count: int | None = None
    total_token_count: int | None = None
    thoughts_token_count: int | None = None
    cached_content_token_count: int | None = None


class GeminiGenerateResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    candidates: tuple[GeminiCandidate, ...] | None = None
    model_version: str | None = None
    response_id: str | None = None
    create_time: str | None = None
    usage_metadata: GeminiUsageMetadata | None = None
    prompt_feedback: dict[str, Any] | None = None
