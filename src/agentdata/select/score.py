"""Difficulty scoring — higher = harder / more reasoning-dense.

Direct port of medqa's `score_difficulty` to DataItem: <think> CoT length, the
presence of a reasoning trace, and user-turn length. Used by both stratified
selection and curriculum ordering. Pure-Python (no numpy) so it stays offline.
"""

from __future__ import annotations

import re

from ..types import DataItem
from ..emit.convert import to_messages

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def score_difficulty(item: DataItem) -> float:
    msgs = to_messages(item)
    asst = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
    user = " ".join(m["content"] for m in msgs if m["role"] == "user")

    score = 0.0
    think = _THINK.search(asst)
    if think:
        cot_words = len(think.group(1).split())
        score += min(cot_words / 100.0, 3.0)
        score += 1.0
    score += min(len(user.split()) / 200.0, 1.0)
    return score
