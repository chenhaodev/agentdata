"""Local file source — JSONL / JSON / CSV under DATA_DIR (the offline test baseline).

Reads ../dataset/{sft_medical.jsonl, pretrain_medical.jsonl, ...} and any
med-data-gen-mvp outputs, detecting each row's format and normalizing to
DataItems. Parquet is supported lazily iff pandas/pyarrow is available.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from ..config import Config
from ..types import DataItem
from ..unify.normalize import normalize_rows

_EXTS = (".jsonl", ".json", ".csv", ".parquet")


class LocalSource:
    name = "local"

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.root = self.config.data_dir

    def _resolve(self, spec: str) -> str:
        """Absolute path: spec as-is if it exists, else relative to DATA_DIR."""
        if os.path.isabs(spec) and os.path.exists(spec):
            return spec
        cand = os.path.join(self.root, spec)
        return cand if os.path.exists(cand) else spec

    def list(self, filters: dict[str, Any] | None = None) -> list[str]:
        if not os.path.isdir(self.root):
            return []
        out = []
        for entry in sorted(os.listdir(self.root)):
            if entry.endswith(_EXTS):
                out.append(entry)
        return out

    def fetch(self, spec: str) -> list[dict[str, Any]]:
        path = self._resolve(spec)
        if not os.path.exists(path):
            raise FileNotFoundError(f"local source: no such file {spec!r} (looked in {self.root!r})")
        ext = os.path.splitext(path)[1].lower()
        if ext == ".jsonl":
            return _read_jsonl(path)
        if ext == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        if ext == ".csv":
            with open(path, encoding="utf-8") as f:
                return list(csv.DictReader(f))
        if ext == ".parquet":
            return _read_parquet(path)
        raise ValueError(f"local source: unsupported extension {ext!r}")

    def to_items(self, raw: list[dict[str, Any]]) -> list[DataItem]:
        return normalize_rows(raw, meta={"source": self.name})

    def load(self, spec: str) -> list[DataItem]:
        rows = self.fetch(spec)
        items = normalize_rows(rows, meta={"source": self.name, "spec": spec})
        return items


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip malformed lines, don't abort the load
    return rows


def _read_parquet(path: str) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # lazy: optional
    except ImportError as e:
        raise ImportError(
            "Reading .parquet needs pandas + pyarrow. Install: pip install pandas pyarrow"
        ) from e
    return pd.read_parquet(path).to_dict(orient="records")
