from __future__ import annotations

import pytest

from inference.providers import openai_compat


@pytest.fixture(autouse=True)
def _reset_pooled_clients():
    """Clear the pooled OpenAI clients around every test.

    The client cache is keyed on construction inputs (base_url/api_key/headers),
    which several tests share while each monkeypatching its own `OpenAI` fake.
    Clearing here guarantees a test always builds against its own patched class
    rather than reusing a client cached by an earlier test.
    """
    openai_compat.reset_client_cache()
    yield
    openai_compat.reset_client_cache()
