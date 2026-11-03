from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class TopLogprob(BaseModel):
    """One candidate token and its log probability at a single position."""

    model_config = ConfigDict(frozen=True)

    token: str
    logprob: float


class TokenLogprob(BaseModel):
    """A sampled output token, its log probability, and the top alternatives."""

    model_config = ConfigDict(frozen=True)

    token: str
    logprob: float
    top_logprobs: tuple[TopLogprob, ...] = ()


def parse_openai_logprobs(raw: Any) -> tuple[TokenLogprob, ...] | None:
    """Convert an OpenAI-compatible `choice.logprobs` into typed `TokenLogprob`s.

    Accepts the live SDK object (attribute access) or a plain dict (from a
    round-tripped payload). Returns None when no per-token content is present.
    """
    if raw is None:
        return None

    content = _get(raw, "content")
    if content is None:
        return None

    tokens: list[TokenLogprob] = []
    for entry in content:
        tokens.append(
            TokenLogprob(
                token=_get(entry, "token"),
                logprob=_get(entry, "logprob"),
                top_logprobs=tuple(
                    TopLogprob(token=_get(alt, "token"), logprob=_get(alt, "logprob"))
                    for alt in (_get(entry, "top_logprobs") or ())
                ),
            )
        )
    return tuple(tokens)


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
