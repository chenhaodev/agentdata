"""Pretrain emitter — {"text": ...} corpus (minimind / nanoGPT continue-pretrain)."""

from __future__ import annotations

from typing import Any

from ..types import DataItem
from .base import BaseEmitter
from .convert import to_text


class PretrainEmitter(BaseEmitter):
    name = "pretrain"
    required = ("text",)

    def row(self, item: DataItem) -> dict[str, Any] | None:
        text = to_text(item).strip()
        if not text:
            return None
        return {"text": text}
