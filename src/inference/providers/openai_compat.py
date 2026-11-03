from __future__ import annotations

import threading
from abc import abstractmethod
from typing import Any, Literal

from openai import OpenAI

from .._dump import dump_provider_response
from ..logprobs import parse_openai_logprobs
from ..prompt import Prompt
from ..provider_responses import OpenAICompletionResponse
from ..response import LLMOutput
from ..settings import InferenceSettings
from ..token_stats import TokenStats
from .base import Model

SystemRole = Literal["system", "developer"]

# A fresh OpenAI() wraps its own httpx connection pool + sockets. Constructing one
# per call under a thread pool leaks file descriptors ([Errno 24] Too many open
# files) and skips keep-alive/TLS reuse. OpenAI/httpx clients are thread-safe to
# share, so we memoize one per distinct construction input (base_url + resolved
# api_key + resolved headers) at module scope — the frozen Model instances can't
# hold it themselves.
_HeaderKey = tuple[tuple[str, str], ...]
_client_cache: dict[tuple[str, str, _HeaderKey], OpenAI] = {}
_client_cache_lock = threading.Lock()


def _make_client(base_url: str, api_key: str, headers: _HeaderKey) -> OpenAI:
    """Return a pooled OpenAI client for these exact inputs, building one on first
    use. Same inputs (from any model/thread) reuse one client; different
    base_url/key/headers get their own, so credentials never cross."""
    key = (base_url, api_key, headers)
    client = _client_cache.get(key)  # fast path: dict get is atomic under the GIL
    if client is not None:
        return client
    with _client_cache_lock:
        # Re-check inside the lock so concurrent first-callers construct only once.
        client = _client_cache.get(key)
        if client is None:
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                default_headers=dict(headers) or None,
            )
            _client_cache[key] = client
        return client


def reset_client_cache() -> None:
    """Drop all pooled clients. Tests that monkeypatch `OpenAI` call this so a
    patched class is rebuilt rather than a stale cached client returned."""
    with _client_cache_lock:
        _client_cache.clear()


class OpenAICompatModel(Model):
    """Shared base for providers that speak OpenAI chat-completions.

    Subclasses supply `_client()` (endpoint + auth) and may reshape the request via
    `_request_kwargs` / `_request_extra_body`.
    """

    base_url: str
    api_key: str | None = None
    system_role: SystemRole = "system"
    extra_body: dict[str, Any] | None = None

    @abstractmethod
    def _client(self) -> OpenAI:
        raise NotImplementedError

    def _request_kwargs(
        self, prompt: Prompt, settings: InferenceSettings
    ) -> dict[str, Any]:
        """Build the chat.completions.create kwargs."""
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
        if settings.logprobs:
            request_kwargs["logprobs"] = True
            if settings.top_logprobs is not None:
                request_kwargs["top_logprobs"] = settings.top_logprobs
        extra_body = self._request_extra_body()
        if extra_body is not None:
            request_kwargs["extra_body"] = extra_body
        return request_kwargs

    def _request_extra_body(self) -> dict[str, Any] | None:
        """Provider-specific fields merged into the JSON request."""
        return dict(self.extra_body) if self.extra_body else None

    def _call(self, prompt: Prompt, settings: InferenceSettings) -> LLMOutput:
        request_kwargs = self._request_kwargs(prompt, settings)
        response = self._client().chat.completions.create(**request_kwargs)
        usage = response.usage
        if usage is None:
            raise ValueError("chat-completions response did not include usage")

        choice = response.choices[0]
        message = choice.message
        completion_details = getattr(usage, "completion_tokens_details", None)
        thinking_tokens = None
        if completion_details is not None:
            thinking_tokens = getattr(completion_details, "reasoning_tokens", None)

        return LLMOutput(
            prompt=prompt,
            text=message.content or "",
            # Some endpoints return `reasoning`; LiteLLM-style gateways (e.g.
            # Stanford's AI API Gateway) return `reasoning_content`.
            thinking=getattr(message, "reasoning", None)
            or getattr(message, "reasoning_content", None),
            tokens=TokenStats(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                thinking_tokens=thinking_tokens,
            ),
            provider_response=OpenAICompletionResponse.model_validate(
                dump_provider_response(response)
            ),
            logprobs=parse_openai_logprobs(getattr(choice, "logprobs", None)),
        )
