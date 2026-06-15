"""Real LLM provider backed by the Anthropic SDK (copied from med-data-gen-mvp).

Used automatically by `get_provider("auto", ...)` when ANTHROPIC_API_KEY is set.
Kept in its own module so importing the package never requires the SDK/key.
"""

from __future__ import annotations

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    is_real = True
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-8") -> None:
        import anthropic  # lazy: only needed when actually used
        self.model = model
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.7) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in msg.content
            if getattr(block, "type", None) == "text"
        )
