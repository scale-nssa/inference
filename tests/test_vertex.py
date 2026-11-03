from __future__ import annotations

import pytest
from anthropic.types import Message
from google.genai.types import GenerateContentResponse
from openai.types.chat import ChatCompletion

from inference import Prompt, llm
from inference.providers.claude_vertex import ClaudeVertexModel
from inference.providers.gemini_vertex import GeminiVertexModel
from inference.providers.gpt_oss_vertex import GptOssVertexModel


def test_claude_call_sends_adaptive_medium_thinking_and_no_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeMessages:
        def create(self, **kwargs: object) -> Message:
            captured["request"] = kwargs
            return Message.model_validate(
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [
                        {"type": "thinking", "thinking": "why", "signature": "sig"},
                        {"type": "text", "text": "answer"},
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            )

    class FakeAnthropicVertex:
        def __init__(self, **kwargs: object) -> None:
            captured["client"] = kwargs
            self.messages = FakeMessages()

    monkeypatch.setattr(
        "inference.providers.claude_vertex.AnthropicVertex", FakeAnthropicVertex
    )

    model = llm.from_name("claude_sonnet_4_6")
    assert isinstance(model, ClaudeVertexModel)
    output = model.call(Prompt(system_prompt="Be terse.", model_prompt="hello"))

    assert captured["client"] == {"project_id": "gcp-nssa-ml", "region": "global"}
    assert captured["request"] == {
        "model": "claude-sonnet-4-6",
        "system": "Be terse.",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8192,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
    }
    assert output.text == "answer"
    assert output.thinking == "why"
    assert output.tokens.input_tokens == 10
    assert output.provider_response.id == "msg_1"


def _patch_anthropic(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    class FakeMessages:
        def create(self, **kwargs: object) -> Message:
            captured["request"] = kwargs
            return Message.model_validate(
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-x",
                    "content": [{"type": "text", "text": "answer"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    class FakeAnthropicVertex:
        def __init__(self, **kwargs: object) -> None:
            self.messages = FakeMessages()

    monkeypatch.setattr(
        "inference.providers.claude_vertex.AnthropicVertex", FakeAnthropicVertex
    )


@pytest.mark.parametrize(("send_temperature", "expected"), [(True, 0.3), (False, None)])
def test_claude_sends_temperature_only_with_thinking_off_and_allowed(
    monkeypatch: pytest.MonkeyPatch, send_temperature: bool, expected: float | None
) -> None:
    captured: dict[str, object] = {}
    _patch_anthropic(monkeypatch, captured)
    model = ClaudeVertexModel(
        nickname="c",
        name="C",
        model_name="claude-x",
        thinking=False,
        ctx_length=1000,
        max_tokens=64,
        send_temperature=send_temperature,
    )

    model.call(Prompt(system_prompt=None, model_prompt="hello"), temperature=0.3)

    assert captured["request"].get("temperature") == expected


def test_claude_call_requires_max_tokens() -> None:
    model = ClaudeVertexModel(
        nickname="c",
        name="C",
        model_name="claude-x",
        thinking=False,
        ctx_length=1000,
    )
    with pytest.raises(ValueError, match="max_tokens"):
        model.call(Prompt(system_prompt=None, model_prompt="hello"))


def test_gemini_call_builds_config_and_splits_thoughts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeModels:
        def generate_content(self, **kwargs: object) -> GenerateContentResponse:
            captured["request"] = kwargs
            return GenerateContentResponse.model_validate(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {"text": "why", "thought": True},
                                    {"text": "answer"},
                                ],
                            },
                            "finish_reason": "STOP",
                        }
                    ],
                    "usage_metadata": {
                        "prompt_token_count": 4,
                        "candidates_token_count": 2,
                        "thoughts_token_count": 3,
                    },
                }
            )

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client"] = kwargs
            self.models = FakeModels()

    monkeypatch.setattr("inference.providers.gemini_vertex.genai.Client", FakeClient)

    model = llm.from_name("gemini_3_5_flash")
    assert isinstance(model, GeminiVertexModel)
    output = model.call(
        Prompt(system_prompt="Be terse.", model_prompt="hello"), max_tokens=256
    )

    assert captured["client"] == {
        "vertexai": True,
        "project": "gcp-nssa-ml",
        "location": "global",
    }
    request = captured["request"]
    assert request["model"] == "gemini-3.5-flash"
    assert request["contents"] == "hello"
    config = request["config"]
    assert config.temperature == 0.0
    assert config.system_instruction == "Be terse."
    assert config.max_output_tokens == 256
    assert config.thinking_config.include_thoughts is True
    assert output.text == "answer"
    assert output.thinking == "why"
    assert output.tokens.thinking_tokens == 3


def test_gpt_oss_call_uses_vertex_endpoint_and_drops_thinking_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> ChatCompletion:
            captured["request"] = kwargs
            return ChatCompletion.model_validate(
                {
                    "id": "c1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "openai/gpt-oss-120b-maas",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "answer",
                                "reasoning_content": "why",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 6,
                        "total_tokens": 15,
                        "completion_tokens_details": {"reasoning_tokens": 4},
                    },
                }
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["client"] = kwargs
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("inference.providers.openai_compat.OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        "inference.providers.gpt_oss_vertex._access_token", lambda: "gcp-token"
    )

    model = llm.from_name("gpt_oss_120b_vertex")
    assert isinstance(model, GptOssVertexModel)
    output = model.call(Prompt(system_prompt=None, model_prompt="hello"))

    assert captured["client"] == {
        "base_url": (
            "https://aiplatform.googleapis.com/v1/projects/gcp-nssa-ml"
            "/locations/global/endpoints/openapi"
        ),
        "api_key": "gcp-token",
        "default_headers": None,
    }
    assert captured["request"] == {
        "model": "openai/gpt-oss-120b-maas",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.0,
        "max_tokens": 16384,
        "extra_body": {"reasoning_effort": "medium"},
    }
    assert output.text == "answer"
    assert output.thinking == "why"
    assert output.tokens.output_tokens == 6
    # Reasoning is already inside completion_tokens; reporting it again double-counts.
    assert output.tokens.thinking_tokens is None
