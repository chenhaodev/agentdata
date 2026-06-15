"""Normalize raw rows → canonical DataItem.

Round-trips the common SFT/chat formats (alpaca ↔ sharegpt ↔ chatml), QA pairs,
and plain text. ShareGPT's {from,value} is mapped to {role,content}; everything
keeps source/tags/lang provenance in DataItem.meta.
"""

from __future__ import annotations

from typing import Any

from ..types import KIND_MESSAGES, KIND_QA, KIND_TEXT, DataItem
from .detect import ALPACA, CHATML, PLAIN, QA, SHAREGPT, detect_format

# ShareGPT role aliases -> canonical chat roles
_SHAREGPT_ROLE = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "observation": "tool",
}


def _norm_messages(raw_msgs: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in raw_msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or _SHAREGPT_ROLE.get((m.get("from") or "").lower(), "user")
        content = m.get("content")
        if content is None:
            content = m.get("value", "")
        out.append({"role": str(role), "content": str(content)})
    return out


def normalize_row(row: dict[str, Any], meta: dict[str, Any] | None = None,
                  fmt: str | None = None) -> DataItem | None:
    """One raw row → one DataItem (or None when the row is empty/unusable)."""
    meta = dict(meta or {})
    fmt = fmt or detect_format(row)

    if fmt in (SHAREGPT, CHATML):
        raw = row.get("conversations") if fmt == SHAREGPT else row.get("messages")
        msgs = _norm_messages(raw or [])
        if not any(m["content"].strip() for m in msgs):
            return None
        return DataItem(kind=KIND_MESSAGES, messages=msgs, meta={**meta, "format": fmt})

    if fmt == ALPACA:
        instr = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        out = (row.get("output") or "").strip()
        user = f"{instr}\n\n{inp}" if inp else instr
        if not user or not out:
            return None
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": out}]
        if row.get("system"):
            msgs.insert(0, {"role": "system", "content": str(row["system"])})
        return DataItem(kind=KIND_MESSAGES, messages=msgs, meta={**meta, "format": fmt})

    if fmt == QA:
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or row.get("output") or "").strip()
        if not q or not a:
            return None
        return DataItem(kind=KIND_QA, question=q, answer=a, meta={**meta, "format": fmt})

    if fmt == PLAIN:
        text = (row.get("text") or "").strip()
        if not text:
            return None
        return DataItem(kind=KIND_TEXT, text=text, meta={**meta, "format": fmt})

    return None


def normalize_rows(rows: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> list[DataItem]:
    items: list[DataItem] = []
    for row in rows:
        it = normalize_row(row, meta)
        if it is not None:
            items.append(it)
    return items
