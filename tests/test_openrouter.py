from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from inference import Prompt, llm
from inference.providers.openrouter import OpenRouterModel


class FakeCompletionDetails(BaseModel):
    model_config = ConfigDict(frozen=True)

    reasoning_tokens: int | None = None


class FakeUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: FakeCompletionDetails | None = None


class FakeMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str = "assistant"
    content: str
    reasoning: str | None = None


class FakeChoice(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int = 0
    finish_reason: str | None = "stop"
    native_finish_reason: str | None = "stop"
    message: FakeMessage


class FakeCompletion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = "gen-test"
    object: str = "chat.completion"
    created: int = 1
    model: str = "anthropic/claude-sonnet-4.5"
    choices: tuple[FakeChoice, ...]
    usage: FakeUsage | None


def test_openrouter_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    model = OpenRouterModel(
        nickname="or",
        name="OpenRouter",
        model_name="openrouter/auto",
        thinking=False,
        ctx_length=200000,
    )

    with pytest.raises(
        ValueError,
        match="OpenRouter requests require api_key or OPENROUTER_API_KEY",
    ):
        model.ask(Prompt(system_prompt=None, model_prompt="hello"))


def test_openrouter_call_uses_openai_compatible_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return FakeCompletion(
                choices=(
                    FakeChoice(
                        message=FakeMessage(
                            content="answer",
                            reasoning="because",
                        )
                    ),
                ),
                usage=FakeUsage(
                    prompt_tokens=11,
                    completion_tokens=7,
                    total_tokens=18,
                    completion_tokens_details=FakeCompletionDetails(
                        reasoning_tokens=3
                    ),
                ),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = type(
                "Chat",
                (),
                {"completions": FakeCompletions()},
            )()

    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.test")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "env-app")
    monkeypatch.setattr("inference.providers.openrouter.OpenAI", FakeOpenAI)
    model = OpenRouterModel(
        nickname="or",
        name="OpenRouter",
        model_name="anthropic/claude-sonnet-4.5",
        thinking=True,
        ctx_length=200000,
        extra_body={"provider": {"only": ["Anthropic"]}},
    )

    output = model.call(
        Prompt(system_prompt="Be terse.", model_prompt="hello"),
        temperature=0.2,
        max_tokens=99,
    )

    assert captured["client"] == {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "env-key",
        "default_headers": {
            "HTTP-Referer": "https://example.test",
            "X-OpenRouter-Title": "env-app",
        },
    }
    assert captured["request"] == {
        "model": "anthropic/claude-sonnet-4.5",
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hello"},
        ],
        "temperature": 0.2,
        "max_tokens": 99,
        "extra_body": {"provider": {"only": ["Anthropic"]}},
    }
    assert output.text == "answer"
    assert output.thinking == "because"
    assert output.tokens.input_tokens == 11
    assert output.tokens.output_tokens == 7
    assert output.tokens.thinking_tokens == 3
    assert output.provider_response.choices[0].native_finish_reason == "stop"


def test_openrouter_catalog_entry_resolves() -> None:
    model = llm.from_name("openrouter_auto", api_key="test-key")

    assert isinstance(model, OpenRouterModel)
    assert model.model_name == "openrouter/auto"
