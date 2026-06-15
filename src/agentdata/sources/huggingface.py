"""HuggingFace source — named-recipe registry + generic dataset ids.

Named recipes resolve to a concrete dataset id, a specific data **file**, and tags
so a Recipe can say `hf:locomo`. The registry ids below were validated live
against the Hub (see tests/test_live.py); the `file`/`format` are pinned because
these datasets don't follow the default split layout that `datasets.load_dataset`
auto-resolves. Generic ids still work: `hf:<dataset_id>` / `hf:<dataset_id>:<split>`.

Downloads prefer `huggingface_hub.hf_hub_download` (auth + cache + LFS), and fall
back to a direct `resolve/main` HTTP fetch for environments where the hub's
Xet/CDN backend is unreachable. `datasets`/`huggingface_hub` are lazy-imported
with install guidance (the agentmem optional-dep pattern).
"""

from __future__ import annotations

import os
import urllib.request
from typing import Any

from ..config import Config
from ..types import DataItem
from ..unify.normalize import normalize_row
from .local import read_records

# named recipes: alias -> resolution. `file` present ⇒ direct-file path; else load_dataset.
REGISTRY: dict[str, dict[str, Any]] = {
    # LoCoMo multiple-choice QA (per-row question_type: single_hop/multi_hop/temporal/...)
    "locomo": {
        "dataset_id": "Percena/locomo-mc10",
        "file": "data/locomo_mc10.json",  # JSONL content despite the .json name
        "format": "jsonl",
        "tags": ["conversational", "long-term-memory"],
        "tag_field": "question_type",  # copy each row's category into its tags
        "license": "see-dataset-card",
    },
    # Claude-Opus distilled reasoning (sharegpt turns carry <think> traces; domain=math)
    "jackrong-claude-opus-distill": {
        "dataset_id": "Jackrong/Claude-opus-4.6-TraceInversion-9000x",
        "file": "claude-opus-4.6-traceInversion-9000x.jsonl",
        "format": "jsonl",
        "tags": ["reasoning", "cot", "distilled", "math"],
        "tag_field": "domain",
        "license": "see-dataset-card",
    },
}

_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"


class HuggingFaceSource:
    name = "huggingface"

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def list(self, filters: dict[str, Any] | None = None) -> list[str]:
        return sorted(REGISTRY.keys())

    def _resolve(self, spec: str) -> dict[str, Any]:
        """Resolve a spec to a registry entry, or a generic {dataset_id, split}."""
        if spec in REGISTRY:
            return dict(REGISTRY[spec])
        dataset_id, _, split = spec.partition(":")
        return {"dataset_id": dataset_id, "split": split or "train", "tags": [], "license": ""}

    def fetch(self, spec: str) -> dict[str, Any]:
        info = self._resolve(spec)
        if info.get("file"):
            path = self._download(info["dataset_id"], info["file"])
            info["rows"] = read_records(path, info.get("format"))
        else:
            info["rows"] = self._load_dataset(info["dataset_id"], info.get("split", "train"))
        return info

    def _download(self, repo: str, file: str) -> str:
        """hf_hub_download first; fall back to a direct resolve-URL fetch."""
        token = self.config.hf_token or None
        try:
            from huggingface_hub import hf_hub_download  # lazy: optional dep

            return hf_hub_download(repo, file, repo_type="dataset", token=token,
                                   cache_dir=self.config.hf_cache_dir or None)
        except ImportError as e:
            raise ImportError(
                "HuggingFace source needs `datasets`/`huggingface_hub`. "
                "Install: pip install 'agentdata[hf]'"
            ) from e
        except Exception:
            return self._http_download(repo, file, token)

    def _http_download(self, repo: str, file: str, token: str | None) -> str:
        dest_dir = os.path.join(self.config.cache_dir, "hf", repo.replace("/", "__"))
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(file))
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return dest  # reuse a good cached file; never re-fetch over it
        headers = {"User-Agent": "agentdata/0.1"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(_RESOLVE.format(repo=repo, file=file), headers=headers)
        # download to a temp file, then atomically swap in — a failed fetch must not
        # clobber a previously-good cache or leave a 0-byte file readers would parse.
        tmp = dest + ".tmp"
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
                f.write(resp.read())
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return dest

    def _load_dataset(self, dataset_id: str, split: str) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset  # lazy: optional dep
        except ImportError as e:
            raise ImportError(
                "HuggingFace source needs `datasets`. Install: pip install 'agentdata[hf]'"
            ) from e
        kwargs: dict[str, Any] = {"split": split}
        if self.config.hf_token:
            kwargs["token"] = self.config.hf_token
        if self.config.hf_cache_dir:
            kwargs["cache_dir"] = self.config.hf_cache_dir
        return [dict(r) for r in load_dataset(dataset_id, **kwargs)]

    def to_items(self, raw: dict[str, Any]) -> list[DataItem]:
        base_meta = {
            "source": self.name,
            "dataset_id": raw.get("dataset_id"),
            "tags": list(raw.get("tags", [])),
            "license": raw.get("license", ""),
        }
        tag_field = raw.get("tag_field")
        items: list[DataItem] = []
        for row in raw.get("rows", []):
            item = normalize_row(row, meta=base_meta)
            if item is None:
                continue
            # per-row category → tags (e.g. LoCoMo question_type, Jackrong domain)
            if tag_field and isinstance(row, dict) and row.get(tag_field):
                item.meta["tags"] = base_meta["tags"] + [str(row[tag_field])]
            items.append(item)
        return items

    def load(self, spec: str) -> list[DataItem]:
        return self.to_items(self.fetch(spec))
