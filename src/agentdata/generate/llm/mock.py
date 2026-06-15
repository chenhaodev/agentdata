"""Deterministic, offline MockProvider (copied from med-data-gen-mvp/src/llm/mock.py).

`is_real == False` so the enrich pass is skipped and template text is used as-is.
`complete` is still implemented (seed-stable) to unit-test the provider contract.
"""

from __future__ import annotations

import hashlib

from .base import LLMProvider


class MockProvider(LLMProvider):
    is_real = False
    name = "mock"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.7) -> str:
        digest = hashlib.sha256(
            f"{self.seed}|{system}|{prompt}".encode("utf-8")
        ).hexdigest()[:12]
        return f"[mock:{digest}]"
