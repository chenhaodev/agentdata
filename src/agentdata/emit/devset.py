"""Devset emitter — a small labeled eval set for prompt/program optimizers.

Prompt optimizers (GEPA, DSPy/MIPROv2, TextGrad, OPRO) don't train weights; they
need a *handful of labeled examples + a metric*. This emitter produces the
labeled half in the shape those tools consume — `{"input", "target"}` rows —
and records a metric stub in the manifest so the dev set is self-describing.

Pair it with a small `Recipe.size` (a dev set is meant to be small + high-signal,
the GEPA principle) to get the few-shot eval split, not a bulk training corpus.
"""

from __future__ import annotations

from typing import Any

from ..types import DataItem, Manifest
from .base import BaseEmitter
from .convert import to_triple

# common exact-match metric optimizers default to; over/under-ridable downstream
DEFAULT_METRIC = "exact_match"


class DevsetEmitter(BaseEmitter):
    name = "devset"
    required = ("input", "target")

    def row(self, item: DataItem) -> dict[str, Any] | None:
        prompt, _inp, target = to_triple(item)
        if not prompt or not target:
            return None
        return {"input": prompt, "target": target}

    def emit(self, items: list[DataItem], path: str) -> Manifest:
        manifest = super().emit(items, path)
        # a dev set is only useful with its scoring rule — record one in the manifest
        manifest.stats["metric"] = DEFAULT_METRIC
        return manifest
