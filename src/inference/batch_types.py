"""Provider-agnostic batch result and status types.

Kept free of provider SDK imports so they can be used (and type-checked) without
pulling in the Google Cloud dependencies that batch.py needs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .prompt import Prompt


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
