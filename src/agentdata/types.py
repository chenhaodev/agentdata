"""Shared data types across sources, unify, diagnose, generate, emit, select.

The canonical currency of the whole package is `DataItem` — every source
normalizes to it and every emitter consumes it. Backend/source specifics live in
`meta` so the common shape stays portable (the agentmem MemoryItem discipline).

Dataclasses (not Pydantic) keep the import path dependency-free and offline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

# canonical item kinds
KIND_MESSAGES = "messages"  # multi-turn chat: meta-agnostic [{role, content}, ...]
KIND_TEXT = "text"  # raw corpus text (pretrain)
KIND_QA = "qa"  # single question/answer pair


@dataclass
class DataItem:
    """One normalized training/eval datum.

    Exactly one shape is populated per `kind`:
      - messages: `messages` = [{"role","content"}, ...]
      - text:     `text` = "..."
      - qa:       `question` + `answer`
    Everything else (source, tags, lang, redistributable, difficulty) is `meta`.
    """

    kind: str  # KIND_MESSAGES | KIND_TEXT | KIND_QA
    messages: list[dict[str, str]] = field(default_factory=list)
    text: str = ""
    question: str = ""
    answer: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def content_key(self) -> str:
        """Stable normalized key for hashing/dedup — the salient user-visible text."""
        if self.kind == KIND_MESSAGES:
            users = [m.get("content", "") for m in self.messages if m.get("role") == "user"]
            return (users[-1] if users else str(self.messages)).strip().lower()
        if self.kind == KIND_TEXT:
            return self.text.strip().lower()
        return f"{self.question}".strip().lower()

    def hash(self) -> str:
        return hashlib.md5(self.content_key().encode("utf-8")).hexdigest()

    def __repr__(self) -> str:  # nicer demo output
        src = self.meta.get("source", "?")
        preview = self.content_key()[:48].replace("\n", " ")
        return f"DataItem(kind={self.kind}, source={src!r}, key={preview!r})"


@dataclass
class Diagnosis:
    """Output of the diagnose stage: which capabilities are weak."""

    scores: dict[str, float] = field(default_factory=dict)  # capability -> [0,1]
    gaps: list[str] = field(default_factory=list)  # below-threshold capabilities
    threshold: float = 0.6
    notes: list[str] = field(default_factory=list)  # human-readable findings


@dataclass
class Recipe:
    """The single contract both the CLI and any future skill compile to.

    `pipeline.run(recipe)` is the only execution path — never two code paths.
    """

    regime: str = "sft"  # pretrain | continue_pretrain | sft | dpo | grpo
    dataset_types: list[str] = field(default_factory=list)  # math|reasoning|domain|intent|multimodal|...
    sources: list[str] = field(default_factory=lambda: ["local"])  # source specs
    emit: str = "sft"  # sft | dpo | pretrain | chat | sharegpt | easydataset
    size: int = 0  # target sample count after select; 0 = keep all
    curriculum: bool = True  # hard-biased stratified select + easy→hard sort
    dedup: bool = True
    generate: dict[str, Any] = field(default_factory=dict)  # synth/recombine/gepa knobs
    out_dir: str = "out"
    name: str = "dataset"  # output subdir / manifest name
    meta: dict[str, Any] = field(default_factory=dict)  # provenance: gaps addressed, etc.

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Recipe":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        if extra:
            kwargs.setdefault("meta", {}).update(extra)
        return cls(**kwargs)


@dataclass
class Manifest:
    """Provenance/audit record returned by emit + the pipeline."""

    name: str
    path: str = ""
    count: int = 0
    emit: str = ""
    sources: list[str] = field(default_factory=list)
    licenses: list[str] = field(default_factory=list)
    gaps_addressed: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "count": self.count,
            "emit": self.emit,
            "sources": self.sources,
            "licenses": self.licenses,
            "gaps_addressed": self.gaps_addressed,
            "stats": self.stats,
        }
