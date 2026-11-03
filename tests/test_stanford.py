from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from inference import Prompt, llm
from inference.providers.stanford import StanfordModel


class FakeMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str = "assistant"
    content: str
    reasoning_content: str | None = None


class FakeChoice(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: FakeMessage
    index: int = 0
    finish_reason: str = "stop"


class FakeUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class FakeCompletion(BaseModel):
    model_config = ConfigDict(frozen=True)

    choices: tuple[FakeChoice, ...]
    usage: FakeUsage
    id: str = "test"
    object: str = "chat.completion"
    created: int = 1
    model: str = "gpt-5.6-luna"


def _model(**overrides: object) -> StanfordModel:
    return StanfordModel(
        nickname="luna",
        name="GPT-5.6 Luna",
        model_name="gpt-5.6-luna",
        thinking=True,
        ctx_length=1_050_000,
        **overrides,
    )


def _patch_openai(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]
) -> None:
    class FakeCompletions:
        def create(self, **kwargs: object) -> FakeCompletion:
            captured["request"] = kwargs
            return FakeCompletion(
                choices=(
                    FakeChoice(
                        message=FakeMessage(content="answer", reasoning_content="why")
                    ),
                ),
                usage=FakeUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["client"] = kwargs
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("inference.providers.openai_compat.OpenAI", FakeOpenAI)


def test_stanford_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STANFORD_AI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="STANFORD_AI_API_KEY"):
        _model().ask(Prompt(system_prompt=None, model_prompt="hello"))


def test_stanford_call_uses_gateway_and_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _patch_openai(monkeypatch, captured)
    monkeypatch.setenv("STANFORD_AI_API_KEY", "stanford-key")

    output = _model(reasoning_effort="high", send_temperature=False).call(
        Prompt(system_prompt="Be terse.", model_prompt="hello"), max_tokens=4000
    )

    assert captured["client"] == {
        "base_url": "https://aiapi-prod.stanford.edu/v1",
        "api_key": "stanford-key",
        "default_headers": None,
    }
    assert captured["request"] == {
        "model": "gpt-5.6-luna",
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hello"},
        ],
        "max_completion_tokens": 4000,
        "extra_body": {"reasoning_effort": "high"},
    }
    assert output.text == "answer"
    assert output.thinking == "why"
    assert output.tokens.input_tokens == 11


def test_stanford_defaults_send_temperature_and_medium_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _patch_openai(monkeypatch, captured)

    _model(api_key="k").call(Prompt(system_prompt=None, model_prompt="hello"))

    assert captured["request"] == {
        "model": "gpt-5.6-luna",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.8,
        "extra_body": {"reasoning_effort": "medium"},
    }


def test_stanford_rejects_reasoning_effort_in_extra_body() -> None:
    with pytest.raises(ValueError, match="reasoning_effort directly"):
        _model(extra_body={"reasoning_effort": "low"})._request_extra_body()


def test_stanford_batch_is_unsupported() -> None:
    with pytest.raises(NotImplementedError, match="no batch endpoint"):
        _model(api_key="k").batch_submit([Prompt(system_prompt=None, model_prompt="x")])


def test_stanford_gpt_5_6_luna_catalog_entry_resolves() -> None:
    model = llm.from_name("stanford_gpt_5_6_luna", api_key="test-key")

    assert isinstance(model, StanfordModel)
    assert model.model_name == "gpt-5.6-luna"
    assert model.reasoning_effort == "medium"
    assert model.send_temperature is False
    assert llm.card("stanford_gpt_5_6_luna").provider == "stanford"


@pytest.mark.parametrize(
    ("nickname", "model_name"),
    [
        ("stanford_gpt_5_6_sol", "gpt-5.6-sol"),
        ("stanford_gpt_5_6_terra", "gpt-5.6-terra"),
        ("stanford_deepseek_v3_2", "DeepSeek-V3.2"),
    ],
)
def test_stanford_catalog_entries_resolve(nickname: str, model_name: str) -> None:
    model = llm.from_name(nickname, api_key="test-key")

    assert isinstance(model, StanfordModel)
    assert model.model_name == model_name
    assert llm.card(nickname).provider == "stanford"
