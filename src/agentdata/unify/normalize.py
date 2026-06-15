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


def _s(v: Any) -> str:
    """Coerce any field value to a stripped string (real corpora carry ints/None/
    floats in answer/text fields — e.g. LoCoMo answers like 2022)."""
    return "" if v is None else str(v).strip()


def _norm_messages(raw_msgs: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in raw_msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or _SHAREGPT_ROLE.get((m.get("from") or "").lower(), "user")
        content = m.get("content")
        if content is None:
            content = m.get("value", "")
        out.append({"role": str(role), "content": "" if content is None else str(content)})
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
        instr = _s(row.get("instruction"))
        inp = _s(row.get("input"))
        out = _s(row.get("output"))
        user = f"{instr}\n\n{inp}" if inp else instr
        if not user or not out:
            return None
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": out}]
        if row.get("system"):
            msgs.insert(0, {"role": "system", "content": _s(row["system"])})
        return DataItem(kind=KIND_MESSAGES, messages=msgs, meta={**meta, "format": fmt})

    if fmt == QA:
        q = _s(row.get("question"))
        a = _s(row.get("answer")) or _s(row.get("output"))
        if not q or not a:
            return None
        return DataItem(kind=KIND_QA, question=q, answer=a, meta={**meta, "format": fmt})

    if fmt == PLAIN:
        text = _s(row.get("text"))
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
