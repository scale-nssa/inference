"""User-facing entrypoints: `llm.from_name(...)` and `llm.call(model, prompt)`."""

from __future__ import annotations

import os
import re
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from .model_card import ModelCard
from .prompt import Prompt
from .providers import PROVIDERS, Model
from .response import LLMOutput

_CATALOG_DIR_ENV = "INFERENCE_CATALOG_PATH"
_BUILTIN_CATALOG = Path(__file__).parent / "catalog"


class UnknownModelError(KeyError):
    pass


class AmbiguousModelError(KeyError):
    pass


@cache
def _catalog() -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}

    dirs = [_BUILTIN_CATALOG]
    extra = os.environ.get(_CATALOG_DIR_ENV)
    if extra:
        dirs.append(Path(extra))

    for d in dirs:
        if not d.is_dir():
            continue
        for yaml_path in sorted(d.glob("*.yaml")):
            entry = yaml.safe_load(yaml_path.read_text())
            nickname = entry["nickname"]
            entries[nickname] = entry  # later dirs override earlier

    return entries


def reload_catalog() -> None:
    """Re-scan the catalog directories from disk."""
    _catalog.cache_clear()


def available() -> list[str]:
    """Return all nicknames in the catalog."""
    return sorted(_catalog())


def from_name(
    name: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **overrides: Any,
) -> Model:
    """Resolve `name` against the catalog and construct the matching provider.

    Matching order:
      1. Exact nickname
      2. Exact model_name
      3. Normalized (strip non-alphanumeric, lowercase, drop @version suffix)
         match against nickname or model_name
      4. Unique normalized-substring match against either
    """
    entry = _resolve(name)
    provider_name = entry["provider"]
    if provider_name not in PROVIDERS:
        raise ValueError(
            f"unknown provider {provider_name!r} in catalog entry "
            f"{entry['nickname']!r}; known: {sorted(PROVIDERS)}"
        )

    init_kwargs = {k: v for k, v in entry.items() if k != "provider"}
    if temperature is not None:
        init_kwargs["temperature"] = temperature
    if max_tokens is not None:
        init_kwargs["max_tokens"] = max_tokens
    init_kwargs.update(overrides)

    return PROVIDERS[provider_name](**init_kwargs)


def call(
    model: Model,
    prompt: Prompt,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMOutput:
    """Call a model and return the full typed response (including provider payload)."""
    return model.call(prompt, temperature=temperature, max_tokens=max_tokens)


def card(name: str) -> ModelCard:
    """Resolve `name` and return its metadata-only :class:`ModelCard`.

    Uses the same matching rules as :func:`from_name` (exact nickname →
    model_name → normalized → unique substring), but reads the catalog only: it
    constructs no provider client, makes no network calls, and needs no API
    credentials. The catalog is the source of truth for model cards.
    """
    return ModelCard.from_entry(_resolve(name))


def cards() -> list[ModelCard]:
    """Return every catalog entry as a :class:`ModelCard`, sorted by nickname."""
    return [ModelCard.from_entry(entry) for _, entry in sorted(_catalog().items())]


def _normalize(s: str) -> str:
    s = re.sub(r"@.*$", "", s)  # strip @version
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _resolve(name: str) -> dict[str, Any]:
    catalog = _catalog()
    if not catalog:
        raise UnknownModelError(
            f"no catalog entries found (looked in {_BUILTIN_CATALOG} "
            f"and ${_CATALOG_DIR_ENV})"
        )

    if name in catalog:
        return catalog[name]

    for entry in catalog.values():
        if entry.get("model_name") == name:
            return entry

    target = _normalize(name)
    exact_norm = [
        e
        for e in catalog.values()
        if _normalize(e["nickname"]) == target
        or _normalize(e.get("model_name", "")) == target
    ]
    if len(exact_norm) == 1:
        return exact_norm[0]
    if len(exact_norm) > 1:
        raise AmbiguousModelError(
            f"{name!r} matches multiple entries: "
            f"{sorted(e['nickname'] for e in exact_norm)}"
        )

    substr = [
        e
        for e in catalog.values()
        if target in _normalize(e["nickname"])
        or target in _normalize(e.get("model_name", ""))
    ]
    if len(substr) == 1:
        return substr[0]
    if len(substr) > 1:
        raise AmbiguousModelError(
            f"{name!r} matches multiple entries: "
            f"{sorted(e['nickname'] for e in substr)}"
        )

    raise UnknownModelError(
        f"no model matching {name!r}. available: {sorted(catalog)}"
    )
