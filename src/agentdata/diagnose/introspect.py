"""Static introspection — scan a target SKILL.md / MCP server / repo for capability
gaps when there is no eval report to parse.

Heuristic, deterministic, offline: walk text files and look for the *absence* of
signals (no reasoning/CoT traces, no tool families, no domain corpus, no temporal
handling). Each missing signal becomes a gap with score 0.0; present signals score
1.0. The output is a Diagnosis usable by the same selector as evalreport.
"""

from __future__ import annotations

import os

from ..types import Diagnosis

# capability -> substrings whose presence (case-insensitive) signals coverage
_SIGNALS: dict[str, tuple[str, ...]] = {
    "reasoning": ("<think>", "reasoning", "chain-of-thought", "step by step", "rationale"),
    "math": ("math", "calculate", "equation", "arithmetic", "compute"),
    "tool_use": ("tool", "function_call", "mcp", "api call", "def "),
    "temporal": ("timeline", "temporal", "date", "history", "over time"),
    "domain": ("domain", "corpus", "knowledge base", "guideline", "reference"),
    "multi_hop": ("multi-hop", "multi_hop", "cross-reference", "combine", "aggregate"),
}

_TEXT_EXTS = (".md", ".py", ".txt", ".json", ".yaml", ".yml", ".toml", ".js", ".ts")
_MAX_BYTES = 2_000_000  # cap how much we read per target


def _read_target(path: str) -> str:
    if os.path.isfile(path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read(_MAX_BYTES)
    chunks: list[str] = []
    budget = _MAX_BYTES
    for root, _dirs, files in os.walk(path):
        for fn in sorted(files):
            if not fn.endswith(_TEXT_EXTS) or budget <= 0:
                continue
            try:
                with open(os.path.join(root, fn), encoding="utf-8", errors="ignore") as f:
                    data = f.read(budget)
            except OSError:
                continue
            chunks.append(data)
            budget -= len(data)
    return "\n".join(chunks)


def scan(path: str, threshold: float = 0.6) -> Diagnosis:
    """Scan a file/dir; capabilities lacking any signal become gaps (score 0.0)."""
    blob = _read_target(path).lower()
    scores: dict[str, float] = {}
    for cap, signals in _SIGNALS.items():
        scores[cap] = 1.0 if any(sig in blob for sig in signals) else 0.0
    gaps = sorted([c for c, s in scores.items() if s < threshold])
    notes = [f"no signal for {c!r} in {os.path.basename(path)}" for c in gaps]
    return Diagnosis(scores=scores, gaps=gaps, threshold=threshold, notes=notes)
