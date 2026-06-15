"""Format detection — guess the schema of a raw row dict.

Lifted from medqa's `detect_format` (train_sft.py) but works on a single row dict
(source-agnostic) rather than a HF Dataset, so the same logic serves files, HF,
and in-memory rows.
"""

from __future__ import annotations

from typing import Any

# canonical format names
ALPACA = "alpaca"
SHAREGPT = "sharegpt"
CHATML = "chatml"
QA = "qa"
PLAIN = "plain"
UNKNOWN = "unknown"


def detect_format(row: dict[str, Any]) -> str:
    """Classify one raw row by its keys (order matters: most specific first)."""
    if not isinstance(row, dict):
        return UNKNOWN
    cols = set(row.keys())
    if "conversations" in cols:
        return SHAREGPT
    if "messages" in cols:
        return CHATML
    if "instruction" in cols:
        return ALPACA
    if "question" in cols and ("answer" in cols or "output" in cols):
        return QA
    if "text" in cols:
        return PLAIN
    return UNKNOWN
