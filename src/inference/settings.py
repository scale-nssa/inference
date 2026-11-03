from pydantic import BaseModel, ConfigDict


class InferenceSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    temperature: float = 0.8
    max_tokens: int | None = None
    structured_output: dict | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
