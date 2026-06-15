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
from .base import collect_provenance
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
        manifest = super().emit(items, path)  # writes the Alpaca JSONL + base manifest
        name = manifest.name
        info = dataset_info_block(name, os.path.basename(path))
        info_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                 f"{name}.dataset_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        manifest.emit = self.name
        manifest.stats["dataset_info_path"] = info_path
        sources, licenses = collect_provenance(items)
        manifest.sources, manifest.licenses = sources, licenses
        return manifest
