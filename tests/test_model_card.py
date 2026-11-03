from __future__ import annotations

import textwrap
from datetime import date

import pytest

from inference import ModelCard, llm
from inference.llm import AmbiguousModelError, UnknownModelError


def test_card_resolves_exact_nickname() -> None:
    card = llm.card("stanford_deepseek_v3_2")
    assert isinstance(card, ModelCard)
    assert card.nickname == "stanford_deepseek_v3_2"
    assert card.model_name == "DeepSeek-V3.2"
    assert card.provider == "stanford"
    assert card.params_b == 671
    assert card.active_params_b == 37
    assert card.pricing == {
        "input": "$0.00 per 1M tokens",
        "output": "$0.00 per 1M tokens",
    }
    assert card.reasoning is False  # non-thinking chat mode
    assert card.logprobs is False


def test_card_matches_from_name_rules() -> None:
    # exact model_name, normalized (drops punctuation/case), and unique substring
    # must all resolve to the same catalog entry that from_name would pick.
    by_model = llm.card("DeepSeek-V3.2")
    by_norm = llm.card("deepseek v3.2")
    by_substr = llm.card("deepseek_v3_2")
    assert (
        by_model.nickname
        == by_norm.nickname
        == by_substr.nickname
        == "stanford_deepseek_v3_2"
    )


def test_card_reasoning_mirrors_thinking() -> None:
    # claude_sonnet_4_6 has thinking: true in the catalog.
    assert llm.card("claude_sonnet_4_6").reasoning is True


@pytest.mark.parametrize(
    ("nickname", "provider", "model_name"),
    [
        ("gemini_3_6_flash", "gemini_vertex", "gemini-3.6-flash"),
        ("gemini_3_5_flash_lite", "gemini_vertex", "gemini-3.5-flash-lite"),
        ("gpt_oss_120b_vertex", "gpt_oss_vertex", "openai/gpt-oss-120b-maas"),
    ],
)
def test_catalog_entries_have_expected_provider_and_model_name(
    nickname: str, provider: str, model_name: str
) -> None:
    card = llm.card(nickname)
    assert card.provider == provider
    assert card.model_name == model_name


def test_card_optional_fields_default_to_none_or_false() -> None:
    # A model with no size/pricing declared still produces a valid card.
    card = llm.card("claude_sonnet_4_6")
    assert card.params_b is None
    assert card.active_params_b is None
    assert card.logprobs is False


def test_card_verified_on_mirrors_verified_key(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "v.yaml").write_text(
        "nickname: verified_test\nname: V\nprovider: stanford\nmodel_name: v\n"
        "thinking: false\nctx_length: 1\nverified: 2026-11-03\n"
    )
    monkeypatch.setenv("INFERENCE_CATALOG_PATH", str(tmp_path))
    llm.reload_catalog()
    try:
        assert llm.card("verified_test").verified_on == date(2026, 11, 3)
    finally:
        monkeypatch.delenv("INFERENCE_CATALOG_PATH", raising=False)
        llm.reload_catalog()


def test_card_is_frozen() -> None:
    card = llm.card("stanford_gpt_5_6_luna")
    with pytest.raises(Exception):
        card.params_b = 999  # type: ignore[misc]


def test_cards_returns_all_entries_sorted() -> None:
    cards = llm.cards()
    nicks = [c.nickname for c in cards]
    assert nicks == sorted(nicks)
    assert set(nicks) == set(llm.available())
    assert all(isinstance(c, ModelCard) for c in cards)


def test_card_needs_no_credentials_or_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """card()/cards() must be pure metadata: no env vars, no provider clients."""
    # Strip every environment variable a provider might read for auth.
    for key in (
        "STANFORD_AI_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
    ):
        monkeypatch.delenv(key, raising=False)

    # Any attempt to construct a provider class would blow up here.
    def _boom(*_args, **_kwargs):
        raise AssertionError("card() must not construct a provider client")

    for name, cls in llm.PROVIDERS.items():
        monkeypatch.setattr(cls, "__init__", _boom)

    # Both a single card and the full listing must still resolve.
    assert llm.card("stanford_gpt_5_6_luna").provider == "stanford"
    assert len(llm.cards()) == len(llm.available())


def test_card_raises_like_from_name() -> None:
    with pytest.raises(UnknownModelError):
        llm.card("no-such-model-xyz")
    with pytest.raises(AmbiguousModelError):
        llm.card("gemini")  # matches many entries


def test_external_catalog_override_produces_card(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INFERENCE_CATALOG_PATH entries are readable as cards too."""
    (tmp_path / "my_model.yaml").write_text(
        textwrap.dedent(
            """
            nickname: card_test_model
            name: Card Test Model
            provider: stanford
            model_name: vendor/card-test
            thinking: false
            ctx_length: 4096
            params_b: 1.5
            pricing:
              input: "$0.01 per 1M tokens"
            supports_logprobs: true
            """
        ).strip()
        + "\n"
    )
    monkeypatch.setenv("INFERENCE_CATALOG_PATH", str(tmp_path))
    llm.reload_catalog()
    try:
        card = llm.card("card_test_model")
        assert card.params_b == 1.5
        assert card.logprobs is True
        assert card.pricing == {"input": "$0.01 per 1M tokens"}
    finally:
        monkeypatch.delenv("INFERENCE_CATALOG_PATH", raising=False)
        llm.reload_catalog()
