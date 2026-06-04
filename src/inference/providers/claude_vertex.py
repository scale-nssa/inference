from anthropic import AnthropicVertex

from .._dump import dump_provider_response
from ..prompt import Prompt
from ..provider_responses import AnthropicMessageResponse
from ..response import LLMOutput
from ..settings import InferenceSettings
from ..token_stats import TokenStats
from .base import Model


class ClaudeVertexModel(Model):
    project_id: str = "gcp-nssa-ml"
    location: str = "global"
    thinking_budget: int = 1024

    def thinking_config(self, max_tokens: int | None) -> dict | None:
        if max_tokens is None:
            raise ValueError("Claude Vertex requests require max_tokens")
        if not self.thinking:
            return None
        if max_tokens <= self.thinking_budget:
            raise ValueError(
                "Claude extended thinking requires max_tokens to be greater than "
                f"thinking_budget; got max_tokens={max_tokens}, "
                f"thinking_budget={self.thinking_budget}"
            )
        return {
            "type": "enabled",
            "budget_tokens": self.thinking_budget,
        }

    def _call(self, prompt: Prompt, settings: InferenceSettings) -> LLMOutput:
        client = AnthropicVertex(project_id=self.project_id, region=self.location)

        request_kwargs: dict = {
            "model": self.model_name,
            "system": prompt.system_prompt or "",
            "messages": [{"role": "user", "content": prompt.model_prompt}],
        }
        if settings.max_tokens is None:
            raise ValueError("Claude Vertex requests require max_tokens")
        request_kwargs["max_tokens"] = settings.max_tokens
        thinking_config = self.thinking_config(settings.max_tokens)
        if thinking_config is not None:
            request_kwargs["thinking"] = thinking_config
        else:
            request_kwargs["temperature"] = settings.temperature

        response = client.messages.create(**request_kwargs)

        cot: str | None = None
        text = ""
        for block in response.content:
            if block.type == "thinking":
                cot = block.thinking
            elif block.type == "text":
                text = block.text

        return LLMOutput(
            prompt=prompt,
            text=text,
            thinking=cot,
            tokens=TokenStats(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            provider_response=AnthropicMessageResponse.model_validate(
                dump_provider_response(response)
            ),
        )
