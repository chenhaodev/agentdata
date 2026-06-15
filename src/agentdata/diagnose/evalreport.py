"""Parse an eval-scores JSON into a Diagnosis.

Accepts three shapes:
  1. a flat capability → score mapping;
  2. a nested LoCoMo-style report ({"scores": {...}} / {"categories": {...}} / per-
     category dicts with an "acc"/"score"/"f1" field);
  3. a **memory-retrieval benchmark** — e.g. the JSON `agentmem/benchmark.py` emits
     ({hit1,hit3,clean,mrr,...} per backend, or a list of such rows). This is how
     `agentdata diagnose --report` evaluates a memory system like agentmem: weak
     retrieval metrics → a `memory` gap the selector answers with LoCoMo-recombined
     long-conversation data.
Scores are coerced to [0,1]; below-threshold = a gap.
"""

from __future__ import annotations

import json
from typing import Any

from ..types import Diagnosis

_SCORE_KEYS = ("score", "acc", "accuracy", "f1", "em", "value")
# retrieval-quality metrics (agentmem benchmark) — fractions in [0,1], higher better.
# non-metric keys (timings, counts, ids) must NOT be read as capability scores.
_MEM_METRICS = ("hit1", "hit3", "hit@1", "hit@3", "clean", "mrr", "recall", "precision")


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


def _is_memory_benchmark(d: dict[str, Any]) -> bool:
    return isinstance(d, dict) and any(k in d for k in _MEM_METRICS)


def _memory_scores(report: "dict | list") -> dict[str, float]:
    """Extract retrieval metrics from a memory benchmark. A list of backend rows is
    reduced to the weakest ok row (lowest mean metric) — the one most in need of data.
    Emits each metric plus a synthesized `memory` capability (their mean)."""
    if isinstance(report, list):
        rows = [r for r in report if isinstance(r, dict) and r.get("ok", True)]
        rows = [r for r in rows if _is_memory_benchmark(r)] or rows
        report = min(rows, key=lambda r: _mean_metric(r)) if rows else {}
    out: dict[str, float] = {}
    for k in _MEM_METRICS:
        if isinstance(report.get(k), (int, float)) and not isinstance(report[k], bool):
            out[k.replace("@", "")] = max(0.0, min(1.0, float(report[k])))
    if out:
        out["memory"] = sum(out.values()) / len(out)  # aggregate retrieval capability
    return out


def _mean_metric(row: dict[str, Any]) -> float:
    vals = [float(row[k]) for k in _MEM_METRICS
            if isinstance(row.get(k), (int, float)) and not isinstance(row[k], bool)]
    return sum(vals) / len(vals) if vals else 1.0


def parse(report: "dict | list", threshold: float = 0.6) -> Diagnosis:
    """Map a scores report → Diagnosis (gaps = capabilities below threshold)."""
    if isinstance(report, list) or _is_memory_benchmark(report):
        scores = _memory_scores(report)
    else:
        scores = _flatten_scores(report)
    gaps = sorted([cap for cap, s in scores.items() if s < threshold], key=lambda c: scores[c])
    notes = [f"{cap}={scores[cap]:.2f} (<{threshold})" for cap in gaps]
    return Diagnosis(scores=scores, gaps=gaps, threshold=threshold, notes=notes)


def parse_file(path: str, threshold: float = 0.6) -> Diagnosis:
    with open(path, encoding="utf-8") as f:
        return parse(json.load(f), threshold=threshold)
