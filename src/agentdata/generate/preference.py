"""Preference-pair generation — attach a `rejected` answer so DPO is self-sufficient.

The DPO emitter needs a `rejected` per item; without one, a DPO recipe writes an
empty file. We synthesize a plausibly-worse answer from the chosen one:
deterministically by **degrading** it (drop the reasoning trace, then truncate to a
terse stub — a less-complete answer is a valid `rejected` for preference training),
and optionally enriched by a real teacher when available. Offline + deterministic
by default so tests and no-key runs still produce real pairs.
"""

from __future__ import annotations

import re

from ..emit.convert import to_messages
from ..types import DataItem
from .llm import LLMProvider, get_provider

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _chosen_answer(item: DataItem) -> str:
    msgs = to_messages(item)
    return next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")


def _degrade(answer: str, max_words: int = 12) -> str:
    """A weaker answer: strip the <think> reasoning, then keep only a terse stub."""
    bare = _THINK.sub("", answer).strip()
    words = bare.split()
    stub = " ".join(words[:max_words])
    return stub if stub and stub != bare else (words[0] if words else "I'm not sure.")


def attach_rejected(items: list[DataItem], provider: LLMProvider | None = None) -> list[DataItem]:
    """Return new items carrying `meta.rejected` (immutably). Items already carrying a
    rejected are left as-is; items without a usable chosen answer are dropped from the
    preference set."""
    provider = provider or get_provider("mock")
    out: list[DataItem] = []
    for it in items:
        if it.meta.get("rejected"):
            out.append(it)
            continue
        chosen = _chosen_answer(it)
        if not chosen.strip():
            continue
        rejected = _degrade(chosen)
        if provider.is_real:  # let a teacher write a more naturally-wrong answer
            try:
                worse = provider.complete(
                    "Write a plausible but clearly worse/less-complete answer to the same "
                    f"question (no reasoning, terse):\n\n{chosen}",
                    max_tokens=200, temperature=0.7)
                rejected = worse.strip() or rejected
            except Exception:
                pass  # fall back to the deterministic degrade
        if rejected.strip() == chosen.strip():
            continue  # no contrast → not a usable pair
        out.append(DataItem(kind=it.kind, messages=list(it.messages), text=it.text,
                            question=it.question, answer=it.answer,
                            meta={**it.meta, "rejected": rejected}))
    return out
