"""Deduplication — drop items with identical normalized content keys.

Reuses medqa's MD5-on-last-user-turn idea, generalized: DataItem.hash() already
hashes the salient content key, so dedup is a stable set-membership pass.
"""

from __future__ import annotations

from ..types import DataItem


def dedup(items: list[DataItem]) -> list[DataItem]:
    seen: set[str] = set()
    out: list[DataItem] = []
    for it in items:
        h = it.hash()
        if h in seen:
            continue
        seen.add(h)
        out.append(it)
    return out
