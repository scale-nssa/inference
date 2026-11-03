"""Metadata-only model "cards".

A :class:`ModelCard` is the intrinsic metadata about a catalog model — size,
pricing, context window, provider, capabilities — projected straight from the
catalog YAML. Cards are built *without* constructing a provider client, so
``llm.card(name)`` resolves and returns a card with no network access and no API
credentials. The catalog (``src/inference/catalog/*.yaml``) is the source of
truth; adding a field here is additive and optional.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict


def _price_per_million(pricing: dict[str, Any] | None, key: str) -> float | None:
    """Parse a per-1M-token USD price out of a catalog pricing string.

    Catalog entries store prices as free text like ``"$0.50 per 1M tokens"``; we pull
    the first numeric value. Returns None when pricing is missing or unparseable.
    """
    if not pricing:
        return None
    value = pricing.get(key)
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None
    match = re.search(r"[-+]?\d*\.?\d+", value.replace(",", ""))
    return float(match.group()) if match else None


class ModelCard(BaseModel):
    """Intrinsic, credential-free metadata for a catalog model.

    Sourced from the catalog entry: ``reasoning`` mirrors the entry's
    ``thinking`` flag, and ``logprobs`` mirrors the entry's ``supports_logprobs``
    capability. ``logprobs`` means *this entry, as configured, returns token-level output
    logprobs through this library*; set it from a live response, not from advertised
    capabilities. ``params_b`` / ``active_params_b``
    are total and (for MoE) active parameters in billions; both are optional so
    partial coverage is fine. ``verified_on`` mirrors the entry's ``verified`` date,
    written by ``scripts/smoke_catalog.py`` when a live call and a price check both
    passed on that day.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    nickname: str
    name: str
    model_name: str
    provider: str
    ctx_length: int
    params_b: float | None = None
    active_params_b: float | None = None
    pricing: dict[str, Any] | None = None
    source_url: str | None = None
    notes: str | None = None
    verified_on: date | None = None
    reasoning: bool = False
    logprobs: bool = False

    @property
    def input_price_per_million(self) -> float | None:
        """USD per 1M input tokens, or None if the catalog has no parseable price."""
        return _price_per_million(self.pricing, "input")

    @property
    def output_price_per_million(self) -> float | None:
        """USD per 1M output tokens, or None if the catalog has no parseable price."""
        return _price_per_million(self.pricing, "output")

    def cost(
        self, input_tokens: int, output_tokens: int = 0, thinking_tokens: int = 0
    ) -> float | None:
        """List-price USD cost for a call, billing thinking tokens at the output rate.

        Reasoning models report thinking tokens separately from the visible answer, but
        providers bill them as output — so they must be included. Returns None if either
        price is missing (can't compute a trustworthy number). Provider/batch discounts
        (e.g. Vertex batch is typically ~50%) are NOT applied here; this is list price.
        """
        p_in = self.input_price_per_million
        p_out = self.output_price_per_million
        if p_in is None or p_out is None:
            return None
        billed_output = (output_tokens or 0) + (thinking_tokens or 0)
        return (input_tokens or 0) / 1e6 * p_in + billed_output / 1e6 * p_out

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> ModelCard:
        """Build a card from a raw catalog entry dict."""
        return cls(
            nickname=entry["nickname"],
            name=entry["name"],
            model_name=entry["model_name"],
            provider=entry["provider"],
            ctx_length=entry["ctx_length"],
            params_b=entry.get("params_b"),
            active_params_b=entry.get("active_params_b"),
            pricing=entry.get("pricing"),
            source_url=entry.get("source_url"),
            notes=entry.get("notes"),
            verified_on=entry.get("verified"),
            reasoning=bool(entry.get("thinking", False)),
            logprobs=bool(entry.get("supports_logprobs", False)),
        )
