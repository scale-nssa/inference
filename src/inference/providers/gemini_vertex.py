from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

from .._dump import dump_provider_response
from ..prompt import Prompt
from ..provider_responses import GeminiGenerateResponse
from ..response import LLMOutput
from ..settings import InferenceSettings
from ..token_stats import TokenStats
from .base import Model


class GeminiVertexModel(Model):
    project_id: str = "gcp-nssa-ml"
    location: str = "global"

    def _call(self, prompt: Prompt, settings: InferenceSettings) -> LLMOutput:
        client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
        )

        config_kwargs = {
            "temperature": settings.temperature,
            "system_instruction": prompt.system_prompt or None,
        }
        if settings.max_tokens is not None:
            config_kwargs["max_output_tokens"] = settings.max_tokens
        config = GenerateContentConfig(**config_kwargs)
        if self.thinking:
            config.thinking_config = ThinkingConfig(include_thoughts=True)

        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt.model_prompt,
            config=config,
        )

        cot: str | None = None
        text = ""
        if self.thinking:
            for part in response.candidates[0].content.parts:
                if part.thought:
                    cot = part.text
                else:
                    text = part.text
        else:
            text = response.text

        usage = response.usage_metadata
        return LLMOutput(
            prompt=prompt,
            text=text,
            thinking=cot,
            tokens=TokenStats(
                input_tokens=usage.prompt_token_count,
                output_tokens=usage.candidates_token_count,
                thinking_tokens=usage.thoughts_token_count,
            ),
            provider_response=GeminiGenerateResponse.model_validate(
                dump_provider_response(response)
            ),
        )
