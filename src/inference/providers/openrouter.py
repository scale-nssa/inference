from __future__ import annotations

import os
from typing import Any, Literal

from openai import OpenAI

from .._dump import dump_provider_response
from ..prompt import Prompt
from ..provider_responses import OpenAICompletionResponse
from ..response import LLMOutput
from ..settings import InferenceSettings
from ..token_stats import TokenStats
from .base import Model

SystemRole = Literal["system", "developer"]

_API_KEY_ENV = "OPENROUTER_API_KEY"
_HTTP_REFERER_ENV = "OPENROUTER_HTTP_REFERER"
_APP_TITLE_ENV = "OPENROUTER_APP_TITLE"


class OpenRouterModel(Model):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str | None = None
    system_role: SystemRole = "system"
    http_referer: str | None = None
    app_title: str | None = "inference"
    extra_body: dict[str, Any] | None = None

    def _client(self) -> OpenAI:
        api_key = self.api_key or os.environ.get(_API_KEY_ENV)
        if not api_key:
            raise ValueError(
                "OpenRouter requests require api_key or OPENROUTER_API_KEY"
            )

        default_headers: dict[str, str] = {}
        http_referer = self._header_value("http_referer", _HTTP_REFERER_ENV)
        app_title = self._header_value("app_title", _APP_TITLE_ENV)
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer
        if app_title:
            default_headers["X-OpenRouter-Title"] = app_title

        return OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            default_headers=default_headers or None,
        )

    def _header_value(self, field_name: str, env_name: str) -> str | None:
        if field_name in self.model_fields_set:
            value = getattr(self, field_name)
            return value if isinstance(value, str) else None
        return os.environ.get(env_name) or getattr(self, field_name)

    def _call(self, prompt: Prompt, settings: InferenceSettings) -> LLMOutput:
        messages: list[dict[str, str]] = []
        if prompt.system_prompt:
            messages.append({"role": self.system_role, "content": prompt.system_prompt})
        messages.append({"role": "user", "content": prompt.model_prompt})

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": settings.temperature,
        }
        if settings.max_tokens is not None:
            request_kwargs["max_tokens"] = settings.max_tokens
        if self.extra_body is not None:
            request_kwargs["extra_body"] = self.extra_body

        response = self._client().chat.completions.create(**request_kwargs)
        usage = response.usage
        if usage is None:
            raise ValueError("OpenRouter response did not include usage")

        message = response.choices[0].message
        completion_details = getattr(usage, "completion_tokens_details", None)
        thinking_tokens = None
        if completion_details is not None:
            thinking_tokens = getattr(completion_details, "reasoning_tokens", None)

        return LLMOutput(
            prompt=prompt,
            text=message.content or "",
            thinking=getattr(message, "reasoning", None),
            tokens=TokenStats(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                thinking_tokens=thinking_tokens,
            ),
            provider_response=OpenAICompletionResponse.model_validate(
                dump_provider_response(response)
            ),
        )
