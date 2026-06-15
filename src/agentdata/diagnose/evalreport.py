"""Parse an eval-scores JSON into a Diagnosis.

Accepts a flat mapping of capability → score, or a nested LoCoMo-style report
({"scores": {...}} / {"categories": {...}} / per-category dicts with an "acc"/
"score"/"f1" field). Scores are coerced to [0,1]; below-threshold = a gap.
"""

from __future__ import annotations

import json
from typing import Any

from ..types import Diagnosis

_SCORE_KEYS = ("score", "acc", "accuracy", "f1", "em", "value")


def _coerce_score(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        s = float(v)
        return s / 100.0 if s > 1.0 else s  # accept 0–100 or 0–1
    if isinstance(v, dict):
        for k in _SCORE_KEYS:
            if k in v:
                return _coerce_score(v[k])
    return None


def _flatten_scores(report: dict[str, Any]) -> dict[str, float]:
    # unwrap a common envelope
    for key in ("scores", "categories", "per_category", "results"):
        if isinstance(report.get(key), dict):
            report = report[key]
            break
    out: dict[str, float] = {}
    for cap, v in report.items():
        s = _coerce_score(v)
        if s is not None:
            out[cap] = max(0.0, min(1.0, s))
    return out


def parse(report: dict[str, Any], threshold: float = 0.6) -> Diagnosis:
    """Map a scores report → Diagnosis (gaps = capabilities below threshold)."""
    scores = _flatten_scores(report)
    gaps = sorted([cap for cap, s in scores.items() if s < threshold], key=lambda c: scores[c])
    notes = [f"{cap}={scores[cap]:.2f} (<{threshold})" for cap in gaps]
    return Diagnosis(scores=scores, gaps=gaps, threshold=threshold, notes=notes)


def parse_file(path: str, threshold: float = 0.6) -> Diagnosis:
    with open(path, encoding="utf-8") as f:
        return parse(json.load(f), threshold=threshold)
