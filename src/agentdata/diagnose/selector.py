"""Selector — the "auto" brain: map diagnosed gaps → a concrete Recipe.

A rule table assigns each capability gap a (regime, dataset_types, sources, emit,
generate) contribution; `select` merges contributions across all gaps, resolves a
single dominant regime by priority, and sets a size budget that bakes in the GEPA
principle (small high-signal SFT + feedback traces beats large RL where rollouts
are costly; verifiable GRPO needs ~500–1000 good pairs, not volume).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..types import Diagnosis, Recipe

# regime priority when multiple gaps disagree (highest wins)
_REGIME_PRIORITY = ["pretrain", "continue_pretrain", "grpo", "dpo", "sft"]

# default size budgets per regime (0 = keep all)
_SIZE = {"pretrain": 0, "continue_pretrain": 0, "grpo": 800, "dpo": 2000, "sft": 1500}


@dataclass
class Rule:
    regime: str
    dataset_types: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    emit: str = "sft"
    generate: dict = field(default_factory=dict)


# gap capability -> Rule (research-grounded mappings; see plan §Diagnosis→Recipe)
RULES: dict[str, Rule] = {
    "temporal": Rule("sft", ["conversational", "temporal"], ["hf:locomo"],
                     "chat", {"recombine": True}),
    "multi_hop": Rule("sft", ["conversational", "multi_hop"], ["hf:locomo"],
                      "chat", {"recombine": True}),
    "single_hop": Rule("sft", ["conversational"], ["hf:locomo"], "chat"),
    "open_domain": Rule("sft", ["domain"], ["local"], "sft"),
    "reasoning": Rule("sft", ["reasoning"], ["hf:jackrong-claude-opus-distill"],
                      "sft", {"gepa": True}),
    "math": Rule("grpo", ["math", "reasoning"], ["hf:jackrong-claude-opus-distill"],
                 "dpo", {"verifiable": True, "gepa": True}),
    "tool_use": Rule("sft", ["intent", "tool"], ["local"], "chat",
                     {"synth": True}),
    "intent": Rule("sft", ["intent"], ["local"], "sft", {"synth": True}),
    "domain": Rule("continue_pretrain", ["domain"], ["local"], "pretrain"),
    "multimodal": Rule("sft", ["multimodal"], ["local"], "chat"),
    # weak retrieval on a memory benchmark (e.g. agentmem) → long-conversation data,
    # recombined into denser multi-session timelines (LoCoMo persona+event method).
    "memory": Rule("sft", ["conversational", "long-term-memory", "temporal", "multi_hop"],
                   ["hf:locomo"], "chat", {"recombine": True}),
    # no multi-agent / role-coordination data → role-conditioned recombined transcripts
    "multi_agent": Rule("sft", ["multi_agent", "conversational"], ["local"], "chat",
                        {"recombine": True, "multi_agent": True}),
}

# fallback when there are no recognized gaps
_DEFAULT = Rule("sft", ["domain"], ["local"], "sft")


def _dominant_regime(regimes: set[str]) -> str:
    for r in _REGIME_PRIORITY:
        if r in regimes:
            return r
    return "sft"


def _dedup_keep_order(seq: list[str]) -> list[str]:
    out: list[str] = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def load_rules(path: str = "") -> dict[str, Rule]:
    """Built-in gap→recipe rules merged with an optional user YAML — so tuning the
    diagnosis brain (or adding a gap) is a fill-in-YAML edit, not a code change. YAML
    is `{gap: {regime, dataset_types, sources, emit, generate}}`; per-gap entries
    override the built-in. See examples/rules.yaml."""
    rules = dict(RULES)
    if path and os.path.exists(path):
        import yaml  # pyyaml is a core dep

        with open(path, encoding="utf-8") as f:
            extra = yaml.safe_load(f) or {}
        for gap, spec in extra.items():
            spec = spec or {}
            rules[gap] = Rule(
                regime=spec.get("regime", "sft"),
                dataset_types=list(spec.get("dataset_types", [])),
                sources=list(spec.get("sources", [])),
                emit=spec.get("emit", "sft"),
                generate=dict(spec.get("generate", {})),
            )
    return rules


def select(diagnosis: Diagnosis, out_dir: str = "out", name: str = "dataset",
           rules: dict[str, Rule] | None = None) -> Recipe:
    """Compose the matched rules into one Recipe."""
    table = rules if rules is not None else RULES
    matched = [table[g] for g in diagnosis.gaps if g in table] or [_DEFAULT]

    regime = _dominant_regime({r.regime for r in matched})
    dataset_types = _dedup_keep_order([t for r in matched for t in r.dataset_types])
    sources = _dedup_keep_order([s for r in matched for s in r.sources])
    generate: dict = {}
    for r in matched:
        generate.update(r.generate)

    # emit: align to the dominant regime, else the first matched rule's emit.
    # GRPO's *data* deliverable is the high-signal SFT set (distilled reasoning);
    # the verifiable-reward/preference pairs are a trainer concern, not data we can
    # honestly fabricate here, so we don't default GRPO to an (empty) DPO export.
    # Explicit DPO regime keeps the DPO export — but only paired sources carrying a
    # `rejected` answer will yield rows.
    emit = {"pretrain": "pretrain", "continue_pretrain": "pretrain",
            "grpo": "sft", "dpo": "dpo"}.get(regime, matched[0].emit)

    return Recipe(
        regime=regime,
        dataset_types=dataset_types,
        sources=sources or ["local"],
        emit=emit,
        size=_SIZE.get(regime, 0),
        curriculum=regime in ("sft", "dpo", "grpo"),
        dedup=True,
        generate=generate,
        out_dir=out_dir,
        name=name,
        meta={"gaps_addressed": list(diagnosis.gaps),
              "diagnosis_scores": dict(diagnosis.scores)},
    )
