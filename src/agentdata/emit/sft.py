"""SFT emitter — {instruction, input, output} JSONL (llm-trainer / LLaMA-Factory alpaca)."""

from __future__ import annotations

from typing import Any

from ..types import DataItem
from .base import BaseEmitter
from .convert import to_triple


class SFTEmitter(BaseEmitter):
    name = "sft"
    required = ("instruction", "output")

    def row(self, item: DataItem) -> dict[str, Any] | None:
        instruction, inp, output = to_triple(item)
        if not instruction or not output:
            return None
        return {"instruction": instruction, "input": inp, "output": output}
