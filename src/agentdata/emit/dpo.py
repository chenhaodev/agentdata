"""DPO emitter — {prompt, chosen, rejected} JSONL (llm-trainer / LLaMA-Factory pairwise).

A preference pair needs a `rejected` answer. Items carry it in `meta["rejected"]`
(attached by generation or by a paired source); items without one are skipped.
"""

from __future__ import annotations

from typing import Any

from ..types import DataItem
from .base import BaseEmitter
from .convert import to_triple


class DPOEmitter(BaseEmitter):
    name = "dpo"
    required = ("prompt", "chosen", "rejected")

    def row(self, item: DataItem) -> dict[str, Any] | None:
        prompt, _, chosen = to_triple(item)
        rejected = (item.meta.get("rejected") or "").strip()
        if not prompt or not chosen or not rejected:
            return None
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
