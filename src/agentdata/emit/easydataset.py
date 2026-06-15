"""easy-dataset emitter — Alpaca JSONL + a LLaMA-Factory `dataset_info` config block.

Matches easy-dataset's one-click LLaMA-Factory export: the data file plus a
`<name>.dataset_info.json` registering it (columns + tags) so LLaMA-Factory can
consume it without hand-editing dataset_info.json.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..types import DataItem, Manifest
from .sft import SFTEmitter


def dataset_info_block(name: str, file_name: str) -> dict[str, Any]:
    """The LLaMA-Factory registry entry for an Alpaca-format file."""
    return {
        name: {
            "file_name": file_name,
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    }


class EasyDatasetEmitter(SFTEmitter):
    name = "easydataset"

    def emit(self, items: list[DataItem], path: str) -> Manifest:
        # super() writes the Alpaca JSONL and a manifest already tagged with
        # emit=self.name + provenance; we just add the LLaMA-Factory config block.
        manifest = super().emit(items, path)
        info = dataset_info_block(manifest.name, os.path.basename(path))
        info_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                 f"{manifest.name}.dataset_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        manifest.stats["dataset_info_path"] = info_path
        return manifest
