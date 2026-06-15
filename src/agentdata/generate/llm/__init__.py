"""Pluggable LLM provider layer (copied in style from med-data-gen-mvp/src/llm).

The generators depend only on the `LLMProvider` ABC, so they run offline and
deterministically with `MockProvider` (tests / no-key) and against a real model
with `AnthropicProvider` when a key is present.
"""

from __future__ import annotations

import os

from .base import LLMProvider
from .mock import MockProvider


def get_provider(provider: str = "auto", model: str = "claude-opus-4-8",
                 seed: int = 0) -> LLMProvider:
    """Resolve a provider name to an instance.

    `auto` -> AnthropicProvider iff ANTHROPIC_API_KEY is set, else MockProvider.
    """
    if provider == "mock":
        return MockProvider(seed=seed)
    if provider == "anthropic" or (provider == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        from .anthropic_provider import AnthropicProvider  # lazy: optional dep/key
        return AnthropicProvider(model=model)
    return MockProvider(seed=seed)


__all__ = ["LLMProvider", "MockProvider", "get_provider"]
