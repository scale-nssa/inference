"""GPT OSS API Service on Vertex AI (Agent Platform), OpenAI-compatible MaaS endpoint.

Vertex serves gpt-oss-120b and gpt-oss-20b as a fully managed service through an
OpenAI-shaped chat-completions endpoint:

    POST https://aiplatform.googleapis.com/v1/projects/{PROJECT}
             /locations/{LOCATION}/endpoints/openapi/chat/completions
    {"model": "openai/gpt-oss-120b-maas", "messages": [...]}

So the REQUEST BODY is OpenAI's and the AUTH is Google's. That split is the whole reason
this is a separate provider rather than a card on StanfordModel: the body can reuse
OpenAICompatModel's `_call`, but the credential is a short-lived GCP access token from
Application Default Credentials, not a static key out of the environment.
"""

from __future__ import annotations

from typing import Any, Literal

import google.auth
import google.auth.transport.requests
from openai import OpenAI
from pydantic import model_validator

from ..prompt import Prompt
from ..response import LLMOutput
from ..settings import InferenceSettings
from ..token_stats import TokenStats
from .openai_compat import OpenAICompatModel, _make_client

ReasoningEffort = Literal["low", "medium", "high"]
_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)

# Module-level so google.auth.default() -- which hits disk and possibly the metadata
# server -- runs once per process rather than once per request. The credential object
# refreshes itself in place, so caching it does not pin an expired token.
_CREDENTIALS = None


def _access_token() -> str:
    """A valid GCP access token from Application Default Credentials.

    Tokens last about an hour, and long runs outlive that, so this refreshes on demand
    rather than once at construction. `_client()` is called per
    request (OpenAICompatModel._call), which is what makes on-demand refresh sufficient.
    """
    global _CREDENTIALS
    if _CREDENTIALS is None:
        _CREDENTIALS, _ = google.auth.default(scopes=list(_SCOPES))
    if not _CREDENTIALS.valid:
        _CREDENTIALS.refresh(google.auth.transport.requests.Request())
    return _CREDENTIALS.token


class GptOssVertexModel(OpenAICompatModel):
    """gpt-oss-* through Vertex's OpenAI-compatible MaaS endpoint.

    Reuses OpenAICompatModel's `_call` for the request/response shape. Three differences:

    - auth is a refreshed GCP access token, not a static API key;
    - base_url is derived from project_id/location, so it is not a settable field;
    - reasoning is a top-level ``reasoning_effort`` (low|medium|high), per the service's
      "Configurable reasoning effort" documentation.
    """

    project_id: str = "gcp-nssa-ml"
    location: str = "global"
    reasoning_effort: ReasoningEffort | None = "medium"
    send_temperature: bool = True

    @model_validator(mode="before")
    @classmethod
    def _derive_base_url(cls, data: Any) -> Any:
        """base_url is DERIVED from project_id/location and is not settable.

        mode="before" because Model is frozen; the value has to be in place at construction.
        """
        if isinstance(data, dict):
            project = data.get("project_id") or "gcp-nssa-ml"
            location = data.get("location") or "global"
            data = dict(data)
            # Regional endpoints live on a regional host; global on the bare one.
            host = (
                "aiplatform.googleapis.com"
                if location == "global"
                else f"{location}-aiplatform.googleapis.com"
            )
            data["base_url"] = (
                f"https://{host}/v1/projects/{project}"
                f"/locations/{location}/endpoints/openapi"
            )
        return data

    def _client(self) -> OpenAI:
        # A refreshed token changes the cache key in _make_client, so each hourly refresh
        # adds one cached client. That is bounded by run length and far cheaper than
        # rebuilding a client per request.
        return _make_client(self.base_url, _access_token(), ())

    def _request_kwargs(
        self, prompt: Prompt, settings: InferenceSettings
    ) -> dict[str, Any]:
        request_kwargs = super()._request_kwargs(prompt, settings)
        if not self.send_temperature:
            del request_kwargs["temperature"]
        return request_kwargs

    def _request_extra_body(self) -> dict[str, Any] | None:
        extra_body = dict(self.extra_body or {})
        if "reasoning_effort" in extra_body:
            raise ValueError(
                "Set GPT OSS Vertex reasoning_effort directly, not through extra_body"
            )
        if self.thinking and self.reasoning_effort is not None:
            extra_body["reasoning_effort"] = self.reasoning_effort
        return extra_body or None

    def _call(self, prompt: Prompt, settings: InferenceSettings) -> LLMOutput:
        """OpenAICompatModel._call, with thinking_tokens deliberately dropped.

        This endpoint counts reasoning inside usage.completion_tokens and also breaks it out
        under usage.completion_tokens_details.reasoning_tokens. The base `_call` copies the
        breakout into TokenStats.thinking_tokens, so ModelCard.cost (input + output +
        thinking) would bill reasoning twice. With thinking_tokens None, cost is input +
        output, which is correct here. The count stays on provider_response and the
        reasoning text on LLMOutput.thinking.
        """
        output = super()._call(prompt, settings)
        if output.tokens.thinking_tokens is None:
            return output
        return output.model_copy(
            update={
                "tokens": TokenStats(
                    input_tokens=output.tokens.input_tokens,
                    output_tokens=output.tokens.output_tokens,
                    thinking_tokens=None,
                )
            }
        )

    # Batch is inherited from base Model (the Vertex BatchPredictionJob path in
    # ../batch, dispatched on _provider_name -> "gpt_oss_vertex").
    #
    # Batch does not currently work: the job is accepted, runs, and then fails at the
    # export stage with "Column url in SELECT * EXCEPT list does not exist" -- the MaaS
    # path appears to want BigQuery I/O rather than gcs_source/gcs_destination. Run this
    # model synchronously until that is resolved.
