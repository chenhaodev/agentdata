"""The common output-emitter contract.

Each Emitter turns canonical DataItems into the exact JSONL schema one trainer
family expects, validates required fields are non-empty strings (llm-trainer
silently drops malformed rows), writes UTF-8 one-object-per-line, and returns a
Manifest. New training targets = new Emitter, no pipeline changes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol, runtime_checkable

from ..types import DataItem, Manifest


@runtime_checkable
class Emitter(Protocol):
    name: str  # the --emit / Recipe.emit key

    def row(self, item: DataItem) -> dict[str, Any] | None:
        """Map one item to a JSONL row, or None to skip it (wrong kind / empty)."""
        ...

    def emit(self, items: list[DataItem], path: str) -> Manifest:
        """Write valid rows to `path` and return a provenance Manifest."""
        ...


def _nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def write_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    """UTF-8, one JSON object per line, parent dirs created."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_provenance(items: list[DataItem]) -> tuple[list[str], list[str]]:
    """Distinct (sources, licenses) from item meta — for the manifest."""
    sources, licenses = [], []
    for it in items:
        s = it.meta.get("source")
        if s and s not in sources:
            sources.append(s)
        lic = it.meta.get("license")
        if lic and lic not in licenses:
            licenses.append(lic)
    return sources, licenses


class BaseEmitter:
    """Shared emit() loop: build rows via `row()`, refuse gated raw data, validate,
    write, and assemble the Manifest. Subclasses implement `row()` + `required`."""

    name: str = "base"
    required: tuple[str, ...] = ()

    def row(self, item: DataItem) -> dict[str, Any] | None:  # pragma: no cover - overridden
        raise NotImplementedError

    def emit(self, items: list[DataItem], path: str) -> Manifest:
        rows: list[dict[str, Any]] = []
        dropped = 0
        gated = 0
        for it in items:
            # honor PhysioNet/MIMIC DUA: never write gated raw data to shareable JSONL
            if it.meta.get("redistributable") is False:
                gated += 1
                continue
            r = self.row(it)
            if r is None:
                dropped += 1
                continue
            if not all(_nonempty_str(_field_text(r, k)) for k in self.required):
                dropped += 1
                continue
            rows.append(r)
        write_jsonl(rows, path)
        sources, licenses = collect_provenance(items)
        return Manifest(
            name=os.path.splitext(os.path.basename(path))[0],
            path=path,
            count=len(rows),
            emit=self.name,
            sources=sources,
            licenses=licenses,
            stats={"input": len(items), "written": len(rows),
                   "dropped": dropped, "gated_skipped": gated},
        )


def _field_text(row: dict[str, Any], key: str) -> Any:
    """Validation helper: for list-valued fields (messages/conversations) check the
    last assistant/value content is a non-empty string; else the field itself."""
    val = row.get(key)
    if isinstance(val, list) and val:
        last = val[-1]
        if isinstance(last, dict):
            return last.get("content") or last.get("value") or ""
        return str(last)
    return val
