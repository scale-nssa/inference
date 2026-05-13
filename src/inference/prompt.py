from pydantic import BaseModel, ConfigDict


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    system_prompt: str | None
    model_prompt: str
