"""LLMProvider ABC (copied in style from med-data-gen-mvp/src/llm/base.py).

The generators build all *structure* deterministically in Python; a real LLM is
used only to ENRICH text into more natural prose (`is_real == True`). With no key
the enrich pass is skipped, so generation stays fully deterministic and offline.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    is_real: bool = False
    name: str = "base"

    @abstractmethod
    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Return a completion for `prompt`."""

    def complete_json(self, prompt: str, *, system: str | None = None,
                      max_tokens: int = 1024, temperature: float = 0.2):
        """Convenience: parse a JSON object/array out of a completion."""
        raw = self.complete(prompt, system=system, max_tokens=max_tokens,
                            temperature=temperature)
        start = min((i for i in (raw.find("{"), raw.find("[")) if i != -1), default=-1)
        if start == -1:
            raise ValueError(f"no JSON found in completion: {raw[:200]!r}")
        return json.loads(raw[start:])
