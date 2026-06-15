"""Kaggle source — datasets via the kaggle API (lazy import + creds).

Credentials come from KAGGLE_USERNAME/KAGGLE_KEY in Config or ~/.kaggle/kaggle.json.
`fetch` downloads + unzips the dataset, then reads any tabular file it finds.
"""

from __future__ import annotations

import glob
import os
from typing import Any

from ..config import Config
from ..types import DataItem
from .local import LocalSource


class KaggleSource:
    name = "kaggle"

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def list(self, filters: dict[str, Any] | None = None) -> list[str]:
        # discovery would need the search API; specs are passed explicitly as <owner>/<dataset>
        return []

    def _client(self):
        if self.config.kaggle_username:
            os.environ.setdefault("KAGGLE_USERNAME", self.config.kaggle_username)
        if self.config.kaggle_key:
            os.environ.setdefault("KAGGLE_KEY", self.config.kaggle_key)
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # lazy: optional dep
        except ImportError as e:
            raise ImportError(
                "Kaggle source needs `kaggle`. Install: pip install 'agentdata[kaggle]' "
                "and set KAGGLE_USERNAME/KAGGLE_KEY (or ~/.kaggle/kaggle.json)."
            ) from e
        api = KaggleApi()
        api.authenticate()
        return api

    def fetch(self, spec: str) -> dict[str, Any]:
        api = self._client()
        dest = os.path.join(self.config.cache_dir, "kaggle", spec.replace("/", "__"))
        os.makedirs(dest, exist_ok=True)
        api.dataset_download_files(spec, path=dest, unzip=True)
        files = [p for ext in ("*.jsonl", "*.json", "*.csv", "*.parquet")
                 for p in glob.glob(os.path.join(dest, "**", ext), recursive=True)]
        return {"spec": spec, "files": files}

    def to_items(self, raw: dict[str, Any]) -> list[DataItem]:
        local = LocalSource(self.config)
        items: list[DataItem] = []
        for path in raw.get("files", []):
            for it in local.load(path):
                it.meta = {**it.meta, "source": self.name, "spec": raw.get("spec")}
                items.append(it)
        return items

    def load(self, spec: str) -> list[DataItem]:
        return self.to_items(self.fetch(spec))
