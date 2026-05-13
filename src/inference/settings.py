from pydantic import BaseModel, ConfigDict


class InferenceSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    temperature: float = 0.8
    max_tokens: int = 8192
    structured_output: dict | None = None
