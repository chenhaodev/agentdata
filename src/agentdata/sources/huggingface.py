"""HuggingFace source — `datasets.load_dataset`, with a named-recipe registry.

Named recipes (LoCoMo, the Jackrong Claude-Opus distilled reasoning set) resolve
to a concrete dataset id + split + tags so a Recipe can say `hf:locomo`. Generic
ids work too: `hf:<dataset_id>` or `hf:<dataset_id>:<split>`.

`datasets` is lazy-imported with install guidance (the agentmem optional-dep
pattern). Network only happens on `fetch`.
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..types import DataItem
from ..unify.normalize import normalize_rows

# named recipes: alias -> (dataset_id, split, tags)
REGISTRY: dict[str, dict[str, Any]] = {
    "locomo": {
        "dataset_id": "snap-research/locomo",
        "split": "train",
        "tags": ["conversational", "long-term-memory", "temporal", "multi_hop"],
        "license": "research-only",
    },
    "jackrong-claude-opus-distill": {
        "dataset_id": "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2",
        "split": "train",
        "tags": ["reasoning", "cot", "distilled"],
        "license": "see-dataset-card",
    },
}


class HuggingFaceSource:
    name = "huggingface"

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def list(self, filters: dict[str, Any] | None = None) -> list[str]:
        return sorted(REGISTRY.keys())

    def _resolve(self, spec: str) -> dict[str, Any]:
        """Resolve a spec to {dataset_id, split, tags, license}.

        Accepts a registry alias ("locomo"), or `dataset_id` / `dataset_id:split`.
        """
        if spec in REGISTRY:
            return dict(REGISTRY[spec])
        dataset_id, _, split = spec.partition(":")
        return {"dataset_id": dataset_id, "split": split or "train", "tags": [], "license": ""}

    def fetch(self, spec: str) -> dict[str, Any]:
        try:
            from datasets import load_dataset  # lazy: optional dep
        except ImportError as e:
            raise ImportError(
                "HuggingFace source needs `datasets`. Install: pip install 'agentdata[hf]'"
            ) from e
        info = self._resolve(spec)
        kwargs: dict[str, Any] = {"split": info["split"]}
        if self.config.hf_token:
            kwargs["token"] = self.config.hf_token
        if self.config.hf_cache_dir:
            kwargs["cache_dir"] = self.config.hf_cache_dir
        ds = load_dataset(info["dataset_id"], **kwargs)
        info["rows"] = [dict(r) for r in ds]
        return info

    def to_items(self, raw: dict[str, Any]) -> list[DataItem]:
        meta = {
            "source": self.name,
            "dataset_id": raw.get("dataset_id"),
            "tags": raw.get("tags", []),
            "license": raw.get("license", ""),
        }
        return normalize_rows(raw.get("rows", []), meta=meta)

    def load(self, spec: str) -> list[DataItem]:
        return self.to_items(self.fetch(spec))
