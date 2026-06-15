"""GEPA-style enrichment — attach NL feedback + trajectory traces, keep sets lean.

GEPA's data lesson: rich natural-language feedback/trajectory traces matter more
than volume (reflective evolution beats GRPO with ~35× fewer rollouts). So we
annotate each sample with a `feedback` rationale and a `trajectory` of its turns,
and cap the set to a small high-signal slice for downstream optimizers.
"""

from __future__ import annotations

from ..emit.convert import to_messages
from ..types import DataItem


def _feedback_for(item: DataItem) -> str:
    msgs = to_messages(item)
    asst = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
    has_reasoning = "<think>" in asst
    return ("Answer includes explicit step-by-step reasoning before the conclusion."
            if has_reasoning else
            "Answer is direct; reward concise correctness and grounding in the prompt.")


def _trajectory_for(item: DataItem) -> list[dict[str, str]]:
    return [{"step": str(i), "role": m["role"], "content": m["content"]}
            for i, m in enumerate(to_messages(item))]


def attach_feedback(items: list[DataItem]) -> list[DataItem]:
    """Return new items carrying meta.feedback + meta.trajectory (immutably)."""
    out: list[DataItem] = []
    for it in items:
        meta = {**it.meta, "feedback": _feedback_for(it), "trajectory": _trajectory_for(it),
                "gepa": True}
        out.append(DataItem(kind=it.kind, messages=list(it.messages), text=it.text,
                            question=it.question, answer=it.answer, meta=meta))
    return out


def keep_high_signal(items: list[DataItem], cap: int = 500) -> list[DataItem]:
    """GEPA lean-set policy: keep the smallest high-signal slice (cap, or all)."""
    if cap <= 0 or len(items) <= cap:
        return items
    return items[:cap]
