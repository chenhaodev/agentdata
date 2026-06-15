"""Shared item→shape conversions used by the emitters.

Keeps every emitter's `row()` tiny: a DataItem of any kind becomes a message list,
an (instruction,input,output) triple, or flat text through one place.
"""

from __future__ import annotations

from ..types import KIND_MESSAGES, KIND_QA, KIND_TEXT, DataItem


def to_messages(item: DataItem) -> list[dict[str, str]]:
    """Any item → a [{role,content}] chat list."""
    if item.kind == KIND_MESSAGES:
        return list(item.messages)
    if item.kind == KIND_QA:
        return [
            {"role": "user", "content": item.question},
            {"role": "assistant", "content": item.answer},
        ]
    if item.kind == KIND_TEXT:
        return [{"role": "assistant", "content": item.text}]
    return []


def to_triple(item: DataItem) -> tuple[str, str, str]:
    """Any item → (instruction, input, output). Empty output ⇒ unusable for SFT."""
    msgs = to_messages(item)
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    users = [m["content"] for m in msgs if m["role"] == "user"]
    assts = [m["content"] for m in msgs if m["role"] == "assistant"]
    instruction = users[0] if users else ""
    if system and instruction:
        instruction = f"{system}\n\n{instruction}"
    output = assts[-1] if assts else ""
    return instruction, "", output


def to_text(item: DataItem) -> str:
    """Any item → flat pretrain text."""
    if item.kind == KIND_TEXT:
        return item.text
    if item.kind == KIND_QA:
        return f"{item.question}\n{item.answer}"
    return "\n".join(f"{m['role']}: {m['content']}" for m in item.messages)
