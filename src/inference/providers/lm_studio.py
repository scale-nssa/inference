"""Unified provider for any OpenAI-compatible local model served by LM Studio.

Subsumes the schematize GPT-OSS / Qwen / Gemma local model classes by varying
two knobs: `system_role` and `thinking_style`.
"""

from __future__ import annotations

import re
from typing import Literal, Self

from openai import OpenAI
from pydantic import model_validator

from .._dump import dump_provider_response
from ..prompt import Prompt
from ..provider_responses import OpenAICompletionResponse
from ..response import LLMOutput
from ..settings import InferenceSettings
from ..token_stats import TokenStats
from ._lm_studio_loader import ensure_model_loaded
from .base import Model

SystemRole = Literal["system", "developer"]
ThinkingStyle = Literal["none", "reasoning_attr", "think_tag"]


class LMStudioModel(Model):
    """Local model served via LM Studio at http://localhost:1234.

    `system_role`: the role label that LM Studio's chat template expects for the
    system prompt. GPT-OSS Harmony uses "developer"; everything else uses "system".

    `thinking_style`: how the model exposes chain-of-thought.
      - "none": no thinking
      - "reasoning_attr": chat-completion message carries a `reasoning` field (gpt-oss)
      - "think_tag": chain-of-thought arrives inline as <think>...</think> (qwen)
    """

    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    system_role: SystemRole = "system"
    thinking_style: ThinkingStyle = "none"
    auto_load: bool = True

    @model_validator(mode="after")
    def _ensure_loaded(self) -> Self:
        if self.auto_load:
            ensure_model_loaded(self.model_name)
        return self

    def _call(self, prompt: Prompt, settings: InferenceSettings) -> LLMOutput:
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        messages: list[dict] = []
        if prompt.system_prompt:
            messages.append({"role": self.system_role, "content": prompt.system_prompt})
        messages.append({"role": "user", "content": prompt.model_prompt})

        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": settings.temperature,
        }
        if settings.max_tokens is not None:
            request_kwargs["max_tokens"] = settings.max_tokens

        response = client.chat.completions.create(**request_kwargs)

        message = response.choices[0].message
        content = message.content or ""

        cot: str | None = None
        text = content

        if self.thinking and self.thinking_style == "reasoning_attr":
            reasoning = getattr(message, "reasoning", None)
            if reasoning:
                cot = reasoning
        elif self.thinking and self.thinking_style == "think_tag":
            match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if match:
                cot = match.group(1).strip()
                text = re.sub(
                    r"<think>.*?</think>", "", content, flags=re.DOTALL
                ).strip()

        return LLMOutput(
            prompt=prompt,
            text=text,
            thinking=cot,
            tokens=TokenStats(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
            provider_response=OpenAICompletionResponse.model_validate(
                dump_provider_response(response)
            ),
        )
