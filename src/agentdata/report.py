"""Provenance / audit report — what was built, from where, under what licenses.

Writes a `<name>.manifest.json` next to the emitted dataset recording sources,
licenses, counts, the recipe, and which diagnosed gaps the build addresses. This
is the audit trail the meta-skill rule requires (esp. for DUA-gated sources).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from .types import DataItem, Manifest, Recipe


def source_breakdown(items: list[DataItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for it in items:
        key = it.meta.get("source", "?")
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_report(recipe: Recipe, items: list[DataItem], manifest: Manifest) -> dict[str, Any]:
    """Assemble the full audit record (manifest + recipe + provenance)."""
    gated = sum(1 for it in items if it.meta.get("redistributable") is False)
    synthetic = sum(1 for it in items if it.meta.get("synthetic"))
    record = {
        **manifest.as_dict(),  # already carries gaps_addressed + provenance fields
        "recipe": asdict(recipe),
        "provenance": {
            "by_source": source_breakdown(items),
            "synthetic_items": synthetic,
            "gated_nonredistributable": gated,
        },
    }
    return record


def write_report(record: dict[str, Any], out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path
