from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Iterable
from typing import Any, Literal

from google.api_core import exceptions as google_exceptions
from google.cloud import aiplatform_v1, storage
from google.cloud.aiplatform_v1.types import (
    BatchPredictionJob,
    GcsDestination,
    GcsSource,
    JobState,
)
from pydantic import BaseModel, ConfigDict

from .prompt import Prompt
from .provider_responses import AnthropicMessageResponse, GeminiGenerateResponse
from .providers.claude_vertex import ClaudeVertexModel
from .providers.gemini_vertex import GeminiVertexModel
from .response import LLMOutput
from .settings import InferenceSettings
from .token_stats import TokenStats

_BATCH_BUCKET_ENV = "INFERENCE_BATCH_GCS_BUCKET"
_BATCH_LOCATION_ENV = "INFERENCE_BATCH_VERTEX_LOCATION"
_REQUEST_LIMIT = 100_000
_TERMINAL_STATES = {
    JobState.JOB_STATE_SUCCEEDED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_PAUSED,
    JobState.JOB_STATE_EXPIRED,
}
_RUNNING_STATES = {
    JobState.JOB_STATE_QUEUED,
    JobState.JOB_STATE_PENDING,
    JobState.JOB_STATE_RUNNING,
    JobState.JOB_STATE_UPDATING,
}
_TRANSIENT_GOOGLE_ERRORS = (
    google_exceptions.DeadlineExceeded,
    google_exceptions.InternalServerError,
    google_exceptions.ServiceUnavailable,
    google_exceptions.TooManyRequests,
)


class BatchError(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt: Prompt
    index: int
    request_id: str
    status: str
    response: dict[str, Any] | None = None


class JobRunning(BaseModel):
    model_config = ConfigDict(frozen=True)

    state: Literal["running"] = "running"
    job_id: str
    provider_state: str
    successful_count: int = 0
    failed_count: int = 0


class JobSucceeded(BaseModel):
    model_config = ConfigDict(frozen=True)

    state: Literal["succeeded"] = "succeeded"
    job_id: str
    provider_state: str
    successful_count: int = 0
    failed_count: int = 0


class JobFailed(BaseModel):
    model_config = ConfigDict(frozen=True)

    state: Literal["failed"] = "failed"
    job_id: str
    provider_state: str
    successful_count: int = 0
    failed_count: int = 0
    error: str | None = None


BatchStatus = JobRunning | JobSucceeded | JobFailed


class BatchJobFailedError(RuntimeError):
    pass


def batch_submit(
    model: ClaudeVertexModel | GeminiVertexModel,
    prompts: list[Prompt],
    *,
    job_name: str | None = None,
) -> str:
    if not prompts:
        raise ValueError("batch_submit requires at least one prompt")
    if len(prompts) > _REQUEST_LIMIT:
        raise ValueError(f"Vertex batch jobs accept at most {_REQUEST_LIMIT} requests")

    settings = model._settings(None, None)
    provider = _provider_name(model)
    location = _batch_location(model)
    job_key = _job_key(job_name)
    input_uri, output_uri = _write_input_jsonl(
        model=model,
        prompts=prompts,
        settings=settings,
        provider=provider,
        job_key=job_key,
    )

    client = _job_client(location)
    job = BatchPredictionJob(
        display_name=job_key,
        model=_publisher_model(provider, model.model_name),
        input_config=BatchPredictionJob.InputConfig(
            instances_format="jsonl",
            gcs_source=GcsSource(uris=[input_uri]),
        ),
        output_config=BatchPredictionJob.OutputConfig(
            predictions_format="jsonl",
            gcs_destination=GcsDestination(output_uri_prefix=output_uri),
        ),
    )
    response = client.create_batch_prediction_job(
        parent=f"projects/{model.project_id}/locations/{location}",
        batch_prediction_job=job,
    )
    return response.name


def batch_status(
    model: ClaudeVertexModel | GeminiVertexModel,
    job_id: str,
) -> BatchStatus:
    job = _get_job(model, job_id)
    state = job.state
    stats = job.completion_stats
    successful_count = int(stats.successful_count or 0)
    failed_count = int(stats.failed_count or 0)
    provider_state = state.name
    if state == JobState.JOB_STATE_SUCCEEDED:
        return JobSucceeded(
            job_id=job.name,
            provider_state=provider_state,
            successful_count=successful_count,
            failed_count=failed_count,
        )
    if state in _RUNNING_STATES or state not in _TERMINAL_STATES:
        return JobRunning(
            job_id=job.name,
            provider_state=provider_state,
            successful_count=successful_count,
            failed_count=failed_count,
        )
    error = str(job.error.message or "") if job.error else None
    return JobFailed(
        job_id=job.name,
        provider_state=provider_state,
        successful_count=successful_count,
        failed_count=failed_count,
        error=error,
    )


def batch_collect(
    model: ClaudeVertexModel | GeminiVertexModel,
    job_id: str,
    prompts: list[Prompt] | None = None,
) -> list[LLMOutput | BatchError]:
    status = batch_status(model, job_id)
    if isinstance(status, JobRunning):
        status = _wait_for_terminal(model, job_id, poll_interval=30.0)
    if isinstance(status, JobFailed):
        raise BatchJobFailedError(
            f"batch job {status.job_id} ended in {status.provider_state}: {status.error}"
        )

    job = _get_job(model, job_id)
    output_uri = job.output_config.gcs_destination.output_uri_prefix
    if prompts is None:
        prompts = _read_input_prompts(job)
    lines = _read_output_jsonl(output_uri)
    provider = _provider_name(model)
    return parse_output_jsonl(provider=provider, prompts=prompts, lines=lines)


def batch(
    model: ClaudeVertexModel | GeminiVertexModel,
    prompts: list[Prompt],
    *,
    job_name: str | None = None,
    poll_interval: float = 30.0,
) -> list[LLMOutput | BatchError]:
    job_id = batch_submit(model, prompts, job_name=job_name)
    status = _wait_for_terminal(model, job_id, poll_interval=poll_interval)
    if isinstance(status, JobFailed):
        raise BatchJobFailedError(
            f"batch job {status.job_id} ended in {status.provider_state}: {status.error}"
        )
    return batch_collect(model, job_id, prompts=prompts)


def format_input_jsonl(
    *,
    provider: Literal["claude_vertex", "gemini_vertex"],
    prompts: list[Prompt],
    model: ClaudeVertexModel | GeminiVertexModel,
    settings: InferenceSettings,
) -> bytes:
    rows = [
        _format_request(
            provider=provider,
            prompt=prompt,
            index=index,
            model=model,
            settings=settings,
        )
        for index, prompt in enumerate(prompts)
    ]
    return ("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n").encode(
        "utf-8"
    )


def parse_output_jsonl(
    *,
    provider: Literal["claude_vertex", "gemini_vertex"],
    prompts: list[Prompt],
    lines: Iterable[str],
) -> list[LLMOutput | BatchError]:
    parsed: dict[int, LLMOutput | BatchError] = {}
    fallback_index = 0
    for raw in lines:
        if not raw.strip():
            continue
        row = json.loads(raw)
        index = _row_index(row, fallback_index)
        fallback_index += 1
        prompt = _prompt_from_row(row, prompts, index)
        request_id = _request_id(index)
        status = row.get("status")
        if status:
            parsed[index] = BatchError(
                prompt=prompt,
                index=index,
                request_id=str(row.get("custom_id") or request_id),
                status=str(status),
                response=row.get("response") if isinstance(row.get("response"), dict) else None,
            )
            continue
        response = row.get("response")
        if not isinstance(response, dict):
            parsed[index] = BatchError(
                prompt=prompt,
                index=index,
                request_id=str(row.get("custom_id") or request_id),
                status="missing response",
                response=None,
            )
            continue
        if provider == "claude_vertex":
            parsed[index] = _anthropic_output(prompt, response)
        elif provider == "gemini_vertex":
            parsed[index] = _gemini_output(prompt, response)
        else:
            raise ValueError(f"unsupported batch provider {provider!r}")

    expected = len(prompts) if prompts else (max(parsed) + 1 if parsed else 0)
    missing = sorted(set(range(expected)) - set(parsed))
    if missing:
        raise ValueError(f"batch output missing request indices: {missing[:10]}")
    return [parsed[index] for index in range(expected)]


def _format_request(
    *,
    provider: Literal["claude_vertex", "gemini_vertex"],
    prompt: Prompt,
    index: int,
    model: ClaudeVertexModel | GeminiVertexModel,
    settings: InferenceSettings,
) -> dict[str, Any]:
    request_id = _request_id(index)
    if provider == "claude_vertex":
        if settings.max_tokens is None:
            raise ValueError("Claude Vertex batch requests require max_tokens")
        request: dict[str, Any] = {
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": settings.max_tokens,
            "system": prompt.system_prompt or "",
            "messages": [{"role": "user", "content": prompt.model_prompt}],
        }
        thinking_config = model.thinking_config(settings.max_tokens)
        if thinking_config is not None:
            request["thinking"] = thinking_config
        else:
            request["temperature"] = settings.temperature
        return {"custom_id": request_id, "request": request}

    if provider == "gemini_vertex":
        generation_config: dict[str, Any] = {
            "temperature": settings.temperature,
        }
        if settings.max_tokens is not None:
            generation_config["maxOutputTokens"] = settings.max_tokens
        if model.thinking:
            generation_config["thinkingConfig"] = {"includeThoughts": True}
        request = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt.model_prompt}],
                }
            ],
            "generationConfig": generation_config,
        }
        if prompt.system_prompt:
            request["systemInstruction"] = {
                "role": "system",
                "parts": [{"text": prompt.system_prompt}],
            }
        return {"custom_id": request_id, "request": request}

    raise ValueError(f"unsupported batch provider {provider!r}")


def _anthropic_output(prompt: Prompt, response: dict[str, Any]) -> LLMOutput:
    provider_response = AnthropicMessageResponse.model_validate(response)
    thinking: str | None = None
    text = ""
    for block in provider_response.content:
        if block.type == "thinking":
            thinking = block.thinking
        elif block.type == "text" and block.text is not None:
            text = block.text
    return LLMOutput(
        prompt=prompt,
        text=text,
        thinking=thinking,
        tokens=TokenStats(
            input_tokens=provider_response.usage.input_tokens,
            output_tokens=provider_response.usage.output_tokens,
        ),
        provider_response=provider_response,
    )


def _gemini_output(prompt: Prompt, response: dict[str, Any]) -> LLMOutput:
    normalized = _camel_to_snake(response)
    provider_response = GeminiGenerateResponse.model_validate(normalized)
    thinking: str | None = None
    text_parts: list[str] = []
    candidates = provider_response.candidates or ()
    if candidates and candidates[0].content and candidates[0].content.parts:
        for part in candidates[0].content.parts:
            if part.thought:
                thinking = part.text
            elif part.text is not None:
                text_parts.append(part.text)
    usage = provider_response.usage_metadata
    return LLMOutput(
        prompt=prompt,
        text="".join(text_parts),
        thinking=thinking,
        tokens=TokenStats(
            input_tokens=(usage.prompt_token_count if usage else None) or 0,
            output_tokens=(usage.candidates_token_count if usage else None) or 0,
            thinking_tokens=usage.thoughts_token_count if usage else None,
        ),
        provider_response=provider_response,
    )


def _write_input_jsonl(
    *,
    model: ClaudeVertexModel | GeminiVertexModel,
    prompts: list[Prompt],
    settings: InferenceSettings,
    provider: Literal["claude_vertex", "gemini_vertex"],
    job_key: str,
) -> tuple[str, str]:
    bucket_name, prefix = _bucket_and_prefix()
    base_prefix = "/".join(part for part in [prefix, job_key] if part)
    input_name = f"{base_prefix}/input/requests.jsonl"
    output_uri = f"gs://{bucket_name}/{base_prefix}/output"
    storage.Client(project=model.project_id).bucket(bucket_name).blob(input_name).upload_from_string(
        format_input_jsonl(provider=provider, prompts=prompts, model=model, settings=settings),
        content_type="application/jsonl",
    )
    return f"gs://{bucket_name}/{input_name}", output_uri


def _read_output_jsonl(output_uri: str) -> list[str]:
    bucket_name, prefix = _parse_gs_uri(output_uri)
    client = storage.Client()
    lines: list[str] = []
    for blob in client.list_blobs(bucket_name, prefix=prefix.rstrip("/") + "/"):
        if not blob.name.endswith(".jsonl"):
            continue
        lines.extend(blob.download_as_text(encoding="utf-8").splitlines())
    if not lines:
        raise FileNotFoundError(f"no JSONL output files found under {output_uri}")
    return lines


def _read_input_prompts(job: BatchPredictionJob) -> list[Prompt]:
    uris = job.input_config.gcs_source.uris
    if not uris:
        return []
    lines = _read_gs_jsonl_uri(uris[0])
    return [_prompt_from_row(json.loads(line), [], index) for index, line in enumerate(lines)]


def _read_gs_jsonl_uri(uri: str) -> list[str]:
    bucket_name, blob_name = _parse_gs_uri(uri)
    text = storage.Client().bucket(bucket_name).blob(blob_name).download_as_text(
        encoding="utf-8"
    )
    return [line for line in text.splitlines() if line.strip()]


def _get_job(model: ClaudeVertexModel | GeminiVertexModel, job_id: str) -> BatchPredictionJob:
    location = _location_from_job_id(job_id) or _batch_location(model)
    return _job_client(location).get_batch_prediction_job(name=job_id)


def _wait_for_terminal(
    model: ClaudeVertexModel | GeminiVertexModel,
    job_id: str,
    *,
    poll_interval: float,
) -> BatchStatus:
    if poll_interval <= 0:
        raise ValueError("poll_interval must be greater than 0")
    delay = max(0.0, poll_interval)
    last_status: BatchStatus | None = None
    while True:
        try:
            status = batch_status(model, job_id)
            last_status = status
        except _TRANSIENT_GOOGLE_ERRORS:
            status = last_status or JobRunning(
                job_id=job_id,
                provider_state="UNKNOWN",
            )
        if not isinstance(status, JobRunning):
            return status
        time.sleep(delay)
        delay = min(delay + poll_interval, 60.0)


def _job_client(location: str) -> aiplatform_v1.JobServiceClient:
    endpoint = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return aiplatform_v1.JobServiceClient(
        client_options={"api_endpoint": endpoint}
    )


def _provider_name(
    model: ClaudeVertexModel | GeminiVertexModel,
) -> Literal["claude_vertex", "gemini_vertex"]:
    if isinstance(model, ClaudeVertexModel):
        return "claude_vertex"
    if isinstance(model, GeminiVertexModel):
        return "gemini_vertex"
    raise NotImplementedError("batch inference is only supported for Vertex Claude and Gemini models")


def _publisher_model(provider: Literal["claude_vertex", "gemini_vertex"], model_name: str) -> str:
    publisher = "anthropic" if provider == "claude_vertex" else "google"
    return f"publishers/{publisher}/models/{model_name}"


def _batch_location(model: ClaudeVertexModel | GeminiVertexModel) -> str:
    override = os.environ.get(_BATCH_LOCATION_ENV)
    if override:
        return override
    if isinstance(model, ClaudeVertexModel) and model.location == "global":
        return "us-east5"
    return model.location


def _bucket_and_prefix() -> tuple[str, str]:
    value = os.environ.get(_BATCH_BUCKET_ENV)
    if not value:
        raise ValueError(f"set ${_BATCH_BUCKET_ENV} to a Cloud Storage bucket or gs:// URI")
    if value.startswith("gs://"):
        return _parse_gs_uri(value)
    bucket, _, prefix = value.partition("/")
    return bucket, prefix.strip("/")


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    match = re.fullmatch(r"gs://([^/]+)(?:/(.*))?", uri)
    if not match:
        raise ValueError(f"expected gs://bucket[/prefix], got {uri!r}")
    return match.group(1), (match.group(2) or "").strip("/")


def _job_key(job_name: str | None) -> str:
    base = job_name or "inference-batch"
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", base).strip("-") or "inference-batch"
    return f"{safe}-{uuid.uuid4().hex[:12]}"


def _request_id(index: int) -> str:
    return f"request-{index:012d}"


def _row_index(row: dict[str, Any], fallback: int) -> int:
    custom_id = row.get("custom_id") or row.get("customId")
    if isinstance(custom_id, str):
        match = re.fullmatch(r"request-(\d+)", custom_id)
        if match:
            return int(match.group(1))
    request = row.get("request")
    if isinstance(request, dict):
        custom_id = request.get("custom_id") or request.get("customId")
        if isinstance(custom_id, str):
            match = re.fullmatch(r"request-(\d+)", custom_id)
            if match:
                return int(match.group(1))
    return fallback


def _prompt_from_row(row: dict[str, Any], prompts: list[Prompt], index: int) -> Prompt:
    if index < len(prompts):
        return prompts[index]
    request = row.get("request")
    if not isinstance(request, dict):
        return Prompt(system_prompt=None, model_prompt="")
    if "messages" in request:
        system = request.get("system") or None
        messages = request.get("messages") or []
        content = messages[0].get("content", "") if messages and isinstance(messages[0], dict) else ""
        return Prompt(system_prompt=system, model_prompt=str(content))
    contents = request.get("contents") or []
    system_instruction = request.get("systemInstruction") or request.get("system_instruction") or {}
    system = _text_from_parts(system_instruction.get("parts") or []) if isinstance(system_instruction, dict) else None
    model_prompt = ""
    if contents and isinstance(contents[0], dict):
        model_prompt = _text_from_parts(contents[0].get("parts") or []) or ""
    return Prompt(system_prompt=system, model_prompt=model_prompt)


def _text_from_parts(parts: list[Any]) -> str | None:
    texts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
    return "".join(texts) or None


def _camel_to_snake(value: Any) -> Any:
    if isinstance(value, list):
        return [_camel_to_snake(item) for item in value]
    if isinstance(value, dict):
        return {_snake_key(key): _camel_to_snake(item) for key, item in value.items()}
    return value


def _snake_key(key: str) -> str:
    return re.sub(r"(?<!^)([A-Z])", r"_\1", key).lower()


def _location_from_job_id(job_id: str) -> str | None:
    match = re.search(r"/locations/([^/]+)/", job_id)
    return match.group(1) if match else None
