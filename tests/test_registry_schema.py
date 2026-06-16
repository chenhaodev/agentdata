"""Offline schema-snapshot tests for the HuggingFace registry. Run: python tests/test_registry_schema.py

Each registered dataset has a frozen representative row in fixtures/registry_samples.json
(captured live, values truncated). These tests re-run detect_format + normalize_row on
those rows with NO network, so a regression in the detector/normalizer — or a registry
entry whose schema we never validated — fails here instead of silently shipping bad data.

Refresh the fixture when adding a dataset (see the builder in the PR that introduced it).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentdata.sources.huggingface import REGISTRY  # noqa: E402
from agentdata.unify.detect import detect_format  # noqa: E402
from agentdata.unify.normalize import normalize_row  # noqa: E402

_FIX = os.path.join(os.path.dirname(__file__), "fixtures", "registry_samples.json")


def _load() -> dict:
    with open(_FIX, encoding="utf-8") as f:
        return json.load(f)


def test_every_registry_entry_has_a_snapshot():
    snaps = _load()
    missing = set(REGISTRY) - set(snaps)
    assert not missing, f"registry datasets without a schema snapshot: {missing}"
    print(f"ok  all {len(REGISTRY)} registry datasets have a frozen schema snapshot")


def test_snapshot_rows_still_detect_and_normalize():
    for alias, snap in _load().items():
        row = snap["sample_row"]
        fmt = detect_format(row)
        assert fmt == snap["expect_format"], (
            f"{alias}: detect_format drifted {snap['expect_format']!r} -> {fmt!r}")
        item = normalize_row(row)
        assert item is not None, f"{alias}: normalize_row now returns None"
        assert item.kind == snap["expect_kind"], (
            f"{alias}: kind drifted {snap['expect_kind']!r} -> {item.kind!r}")
    print(f"ok  {len(_load())} snapshot rows still detect + normalize to the same shape")


def test_snapshot_dataset_ids_match_registry():
    snaps = _load()
    for alias, snap in snaps.items():
        if alias in REGISTRY:
            assert snap["dataset_id"] == REGISTRY[alias]["dataset_id"], (
                f"{alias}: snapshot dataset_id != registry dataset_id")
    print("ok  snapshot dataset_ids match the registry")


def test_tag_field_entries_carry_per_row_category():
    # hermes uses tag_field=category; the snapshot row must actually have that field,
    # else the per-row tagging silently no-ops.
    snaps = _load()
    for alias, info in REGISTRY.items():
        tf = info.get("tag_field")
        if tf and alias in snaps:
            assert tf in snaps[alias]["sample_row"], (
                f"{alias}: tag_field {tf!r} absent from sample row")
    print("ok  tag_field datasets have their category field present")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    main()
