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
import tempfile
import urllib.request
from typing import Any

from ..config import Config
from ..types import DataItem
from ..unify.normalize import normalize_row
from .local import read_records

# named recipes: alias -> resolution. `file` present ⇒ direct-file path; else load_dataset.
# `likes`/`downloads` snapshot the Hub popularity at registration (2026-06-15): popularity
# is a real quality signal, so registry picks are the most-liked option that also unifies
# cleanly. Use `HuggingFaceSource.search(q)` to re-rank live before adding new recipes.
REGISTRY: dict[str, dict[str, Any]] = {
    # LoCoMo multiple-choice QA (per-row question_type: single_hop/multi_hop/temporal/...).
    # The canonical LoCoMo data lives in the GitHub repo, so all HF mirrors are low-like;
    # this is the most-liked mirror that ships a clean unifiable file.
    "locomo": {
        "dataset_id": "Percena/locomo-mc10",
        "file": "data/locomo_mc10.json",  # JSONL content despite the .json name
        "format": "jsonl",
        "tags": ["conversational", "long-term-memory"],
        "tag_field": "question_type",  # copy each row's category into its tags
        "license": "see-dataset-card",
        "likes": 7, "downloads": 970,
    },
    # Claude-Opus distilled reasoning (sharegpt turns carry <think> traces; domain=math).
    # The most-liked Claude-distilled reasoning set on the Hub.
    "jackrong-claude-opus-distill": {
        "dataset_id": "Jackrong/Claude-opus-4.6-TraceInversion-9000x",
        "file": "claude-opus-4.6-traceInversion-9000x.jsonl",
        "format": "jsonl",
        "tags": ["reasoning", "cot", "distilled", "math"],
        "tag_field": "domain",
        "license": "see-dataset-card",
        "likes": 69, "downloads": 1902,
    },
    # Multi-turn agent trajectories (ETO SFT: AlfWorld/SciWorld/WebShop tool-use turns).
    # sharegpt {from,value} inside a JSON array → unifies directly. The most-liked
    # agent-trajectory set on the Hub; backs the tool/trajectory optimization row.
    "agent-trajectory": {
        "dataset_id": "agent-eto/eto-sft-trajectory",
        "file": "data/alfworld_sft.json",
        "format": "json",
        "tags": ["agent", "trajectory", "tool-use"],
        "license": "see-dataset-card",
        "likes": 17, "downloads": 224,
    },
    # Cleaned Alpaca instruction-tuning set (alpaca {instruction,input,output}); the
    # de-facto general SFT baseline and the most-liked alpaca variant on the Hub.
    # Validated 2026-06-16: alpaca_data_cleaned.json → 51,760 rows, 500/500 unify.
    "alpaca-cleaned": {
        "dataset_id": "yahma/alpaca-cleaned",
        "file": "alpaca_data_cleaned.json",
        "format": "json",
        "tags": ["instruction", "sft", "general"],
        "license": "cc-by-4.0",
        "likes": 841, "downloads": 22143,
    },
    # Hermes function-calling (sharegpt conversations with system+tool roles and a
    # <tools> spec); backs the tool/function-calling optimization row. The most-liked
    # function-calling set that unifies as-is. Validated 2026-06-16: 1,893 rows.
    "hermes-function-calling": {
        "dataset_id": "NousResearch/hermes-function-calling-v1",
        "file": "func-calling-singleturn.json",
        "format": "json",
        "tags": ["agent", "tool-use", "function-calling"],
        "tag_field": "category",  # per-row task category → tags
        "license": "apache-2.0",
        "likes": 421, "downloads": 30541,
    },
}

_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"


def load_registry(path: str = "") -> dict[str, dict[str, Any]]:
    """Built-in registry merged with an optional user YAML — so adding a dataset
    recipe is a fill-in-YAML edit, not a code change. The YAML is `{alias: {dataset_id,
    file, format, tags, ...}}`; per-alias keys override the built-in. See
    examples/registry.yaml."""
    merged = {alias: dict(entry) for alias, entry in REGISTRY.items()}
    if path and os.path.exists(path):
        import yaml  # pyyaml is a core dep

        with open(path, encoding="utf-8") as f:
            extra = yaml.safe_load(f) or {}
        for alias, entry in extra.items():
            merged[alias] = {**merged.get(alias, {}), **(entry or {})}
    return merged


class HuggingFaceSource:
    name = "huggingface"

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.registry = load_registry(self.config.registry_path)

    def list(self, filters: dict[str, Any] | None = None) -> list[str]:
        return sorted(self.registry.keys())

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Discover Hub datasets ranked by popularity (likes). Likes/downloads are a
        real dataset-quality signal — surface them so a human (or a recipe author)
        picks vetted data instead of the first string match."""
        try:
            from huggingface_hub import HfApi  # lazy: optional dep
        except ImportError as e:
            raise ImportError(
                "HuggingFace search needs `huggingface_hub`. Install: pip install 'agentdata[hf]'"
            ) from e
        api = HfApi(token=self.config.hf_token or None)
        out = []
        for d in api.list_datasets(search=query, sort="likes", limit=limit):
            out.append({"id": d.id,
                        "likes": getattr(d, "likes", 0) or 0,
                        "downloads": getattr(d, "downloads", 0) or 0})
        return out

    def _resolve(self, spec: str) -> dict[str, Any]:
        """Resolve a spec to a registry entry, or a generic {dataset_id, split}."""
        if spec in self.registry:
            return dict(self.registry[spec])
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
        # download to a unique temp file, then atomically swap in — a failed fetch must
        # not clobber a good cache or leave a 0-byte file, and concurrent downloads of
        # the same spec (parallel router) must not race on a shared temp path.
        fd, tmp = tempfile.mkstemp(dir=dest_dir, suffix=".tmp")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                os.write(fd, resp.read())
            os.close(fd)
            fd = -1  # closed; don't double-close in finally
            os.replace(tmp, dest)
        finally:
            if fd != -1:
                os.close(fd)
            if os.path.exists(tmp):  # only if replace didn't happen
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
