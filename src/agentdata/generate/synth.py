"""Synthetic QA/CoT generation from seed docs (distilabel-style teacher gen).

Deterministic structure first: from each KIND_TEXT seed we build a grounded
question + a `<think>`-wrapped reasoning answer from the seed content. When a real
provider is available the prose is enriched; otherwise the template output is used
verbatim so generation stays offline and reproducible.
"""

from __future__ import annotations

from ..types import KIND_MESSAGES, KIND_TEXT, DataItem
from .llm import LLMProvider, get_provider

_SYS = "You are a teacher creating high-signal training examples with explicit reasoning."


def _seed_text(item: DataItem) -> str:
    if item.kind == KIND_TEXT:
        return item.text
    if item.kind == KIND_MESSAGES:
        return " ".join(m.get("content", "") for m in item.messages)
    return f"{item.question} {item.answer}"


def _first_sentences(text: str, n: int = 2) -> str:
    parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
    return ". ".join(parts[:n]) + ("." if parts else "")


def synth_one(seed: DataItem, provider: LLMProvider, reasoning: bool = True) -> DataItem | None:
    text = _seed_text(seed).strip()
    if len(text.split()) < 8:
        return None
    topic = _first_sentences(text, 1)
    question = f"Explain and reason about the following: {topic}"
    body = _first_sentences(text, 3)
    if reasoning:
        answer = f"<think>\nThe passage states: {body}\nI work through what it implies step by step.\n</think>\n{body}"
    else:
        answer = body

    if provider.is_real:  # enrich the deterministic skeleton into natural prose
        try:
            answer = provider.complete(
                f"Rewrite this as a clear reasoned answer, keep any <think> block:\n\n{answer}",
                system=_SYS, max_tokens=800, temperature=0.5,
            ) or answer
        except Exception:
            pass  # fall back to the deterministic skeleton

    return DataItem(
        kind=KIND_MESSAGES,
        messages=[{"role": "user", "content": question},
                  {"role": "assistant", "content": answer}],
        meta={**seed.meta, "synthetic": True, "gen": "synth",
              "source": seed.meta.get("source", "synth")},
    )


def synth(seeds: list[DataItem], provider: LLMProvider | None = None,
          reasoning: bool = True, limit: int = 0) -> list[DataItem]:
    """Generate one reasoned QA item per usable seed (capped by `limit` if > 0)."""
    provider = provider or get_provider("mock")
    out: list[DataItem] = []
    for seed in seeds:
        item = synth_one(seed, provider, reasoning=reasoning)
        if item is not None:
            out.append(item)
        if limit and len(out) >= limit:
            break
    return out
