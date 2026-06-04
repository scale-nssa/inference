from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from ..prompt import Prompt
from ..response import LLMOutput, Response
from ..settings import InferenceSettings


class Model(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)

    nickname: str
    name: str
    model_name: str
    thinking: bool
    ctx_length: int

    temperature: float = 0.8
    max_tokens: int | None = None

    def _settings(
        self,
        temperature: float | None,
        max_tokens: int | None,
    ) -> InferenceSettings:
        return InferenceSettings(
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )

    def ask(
        self,
        prompt: Prompt,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Response:
        return self.call(
            prompt, temperature=temperature, max_tokens=max_tokens
        ).to_response()

    def call(
        self,
        prompt: Prompt,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMOutput:
        return self._call(prompt, self._settings(temperature, max_tokens))

    def batch_submit(
        self,
        prompts: list[Prompt],
        *,
        job_name: str | None = None,
    ) -> str:
        from ..batch import batch_submit

        return batch_submit(self, prompts, job_name=job_name)

    def batch_status(self, job_id: str):
        from ..batch import batch_status

        return batch_status(self, job_id)

    def batch_collect(self, job_id: str):
        from ..batch import batch_collect

        return batch_collect(self, job_id)

    def batch(
        self,
        prompts: list[Prompt],
        *,
        job_name: str | None = None,
        poll_interval: float = 30.0,
    ):
        from ..batch import batch

        return batch(
            self,
            prompts,
            job_name=job_name,
            poll_interval=poll_interval,
        )

    @abstractmethod
    def _call(self, prompt: Prompt, settings: InferenceSettings) -> LLMOutput:
        raise NotImplementedError
