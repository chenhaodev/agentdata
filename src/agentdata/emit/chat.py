"""Chat emitters — ChatML {"messages":[...]} and ShareGPT {"conversations":[...]}.

ChatML feeds llm-trainer's SFTDataset (conversations/messages of {role,content});
ShareGPT uses {from,value} for LoCoMo-style multi-turn tooling.
"""

from __future__ import annotations

from typing import Any

from ..types import DataItem
from .base import BaseEmitter
from .convert import to_messages

# ChatML role -> ShareGPT `from`
_SHAREGPT_FROM = {"user": "human", "assistant": "gpt", "system": "system", "tool": "observation"}


class ChatEmitter(BaseEmitter):
    name = "chat"
    required = ("messages",)

    def row(self, item: DataItem) -> dict[str, Any] | None:
        msgs = [m for m in to_messages(item) if m.get("content", "").strip()]
        if len(msgs) < 2:
            return None
        return {"messages": msgs}


class ShareGPTEmitter(BaseEmitter):
    name = "sharegpt"
    required = ("conversations",)

    def row(self, item: DataItem) -> dict[str, Any] | None:
        msgs = [m for m in to_messages(item) if m.get("content", "").strip()]
        if len(msgs) < 2:
            return None
        convs = [{"from": _SHAREGPT_FROM.get(m["role"], "human"), "value": m["content"]}
                 for m in msgs]
        return {"conversations": convs}
