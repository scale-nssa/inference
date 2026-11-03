"""Stanford AI API Gateway provider."""

from __future__ import annotations

import os
from typing import Any, Literal

from openai import OpenAI

from ..prompt import Prompt
from ..settings import InferenceSettings
from .openai_compat import OpenAICompatModel, _make_client

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class StanfordModel(OpenAICompatModel):
    """Stanford's AI API Gateway through its OpenAI-compatible chat-completions
    endpoint. Authenticates with ``Authorization: Bearer <key>`` from
    STANFORD_AI_API_KEY; each key only reaches the models it was granted for.

    Reuses OpenAICompatModel's `_call`; the request differs in three ways:

    - reasoning is a top-level ``reasoning_effort`` field (the gateway's documented
      reasoning payload);
    - output length goes out as ``max_completion_tokens``, which OpenAI reasoning
      models require in place of ``max_tokens``;
    - ``send_temperature: false`` omits ``temperature``, since OpenAI reasoning
      models reject any value other than the default.
    """

    base_url: str = "https://aiapi-prod.stanford.edu/v1"
    reasoning_effort: ReasoningEffort | None = "medium"
    send_temperature: bool = True

    def _client(self) -> OpenAI:
        api_key = self.api_key or os.environ.get("STANFORD_AI_API_KEY")
        if not api_key:
            raise ValueError(
                "Stanford AI API Gateway requests require api_key or "
                "STANFORD_AI_API_KEY"
            )
        return _make_client(self.base_url, api_key, ())

    def _request_kwargs(
        self, prompt: Prompt, settings: InferenceSettings
    ) -> dict[str, Any]:
        request_kwargs = super()._request_kwargs(prompt, settings)
        if not self.send_temperature:
            del request_kwargs["temperature"]
        if "max_tokens" in request_kwargs:
            request_kwargs["max_completion_tokens"] = request_kwargs.pop("max_tokens")
        return request_kwargs

    def _request_extra_body(self) -> dict[str, Any] | None:
        extra_body = dict(self.extra_body or {})
        if "reasoning_effort" in extra_body:
            raise ValueError(
                "Set Stanford reasoning_effort directly, not through extra_body"
            )
        if self.thinking and self.reasoning_effort is not None:
            extra_body["reasoning_effort"] = self.reasoning_effort
        return extra_body or None

    # The gateway has no batch endpoint; fail clearly instead of falling through to
    # the base Model's Vertex batch path.

    def batch_submit(self, prompts, *, job_name=None):
        raise NotImplementedError("Stanford AI API Gateway has no batch endpoint")

    def batch_status(self, job_id):
        raise NotImplementedError("Stanford AI API Gateway has no batch endpoint")

    def batch_collect(self, job_id):
        raise NotImplementedError("Stanford AI API Gateway has no batch endpoint")

    def batch(self, prompts, *, job_name=None, poll_interval=30.0):
        raise NotImplementedError("Stanford AI API Gateway has no batch endpoint")
