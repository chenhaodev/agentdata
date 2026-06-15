"""The common input-source contract.

Every connector (local files, HuggingFace, Kaggle, PhysioNet, GitHub) is wrapped
to satisfy this Protocol so adding a source never touches the pipeline. The shape
is the lowest common denominator: discover specs, fetch raw, normalize to items.
Source-specific superpowers ride in DataItem.meta, not in this contract — the
exact agentmem LongTermBackend discipline.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..types import DataItem


@runtime_checkable
class DataSource(Protocol):
    name: str

    def list(self, filters: dict[str, Any] | None = None) -> list[str]:
        """Discover available specs (dataset ids / file paths) for this source."""
        ...

    def fetch(self, spec: str) -> Any:
        """Pull the raw payload for one spec (rows, file handle, repo dir, ...)."""
        ...

    def to_items(self, raw: Any) -> list[DataItem]:
        """Normalize a raw payload into canonical DataItems (tagged with provenance)."""
        ...

    def load(self, spec: str) -> list[DataItem]:
        """Convenience: fetch + to_items for one spec."""
        ...
