from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TokenStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int
    thinking_tokens: int | None = None

    def __add__(self, other: "TokenStats") -> "TokenStats":
        def _add_opt(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return TokenStats(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            thinking_tokens=_add_opt(self.thinking_tokens, other.thinking_tokens),
        )
