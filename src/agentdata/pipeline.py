"""Pipeline core — the single entry point the CLI and any skill call.

`run(recipe)` executes every stage and writes the dataset + manifest.
`run_stage(stage, recipe)` runs deterministically up to one stage and returns its
output, so any stage can be inspected/re-run independently (the med-data-gen rule).

Stages: load → generate → dedup → select → emit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .config import Config
from .emit import build_emitter
from .generate import (
    attach_feedback, attach_rejected, get_provider, keep_high_signal, recombine, synth,
)
from .report import build_report, write_report
from .select import curriculum_select, dedup
from .sources import load_sources
from .types import DataItem, Manifest, Recipe

STAGES = ["load", "generate", "dedup", "select", "emit"]


@dataclass
class RunResult:
    manifest: Manifest
    report_path: str
    report: dict[str, Any]


def _wants_reasoning(recipe: Recipe) -> bool:
    return "reasoning" in recipe.dataset_types or recipe.regime in ("grpo", "dpo")


# -- individual stages -------------------------------------------------------

def stage_load(recipe: Recipe, config: Config) -> list[DataItem]:
    return load_sources(recipe.sources, config)


def stage_generate(items: list[DataItem], recipe: Recipe, config: Config) -> list[DataItem]:
    """Apply synth / recombine / preference / gepa. Additive: generated items extend
    the loaded pool; preference attaches a `rejected` so DPO is self-sufficient; gepa
    annotates the whole pool. Runs whenever a generator is requested *or* the emit
    format needs it (DPO)."""
    gen = recipe.generate or {}
    # a DPO export needs preference pairs even if the recipe set no generate flags
    want_preference = bool(gen.get("preference")) or recipe.emit == "dpo"
    if not gen and not want_preference:
        return items
    out = list(items)
    if gen.get("recombine") or gen.get("synth"):
        provider = get_provider(config.llm_provider, config.llm_model)  # built only when a generator needs it
        if gen.get("recombine"):
            out += recombine(items, provider, multi_agent=bool(gen.get("multi_agent")))
        if gen.get("synth"):
            out += synth(items, provider, reasoning=_wants_reasoning(recipe))
    if want_preference:
        out = attach_rejected(out, get_provider(config.llm_provider, config.llm_model))
    if gen.get("gepa"):
        out = attach_feedback(out)
        out = keep_high_signal(out, cap=int(gen.get("gepa_cap", 500)))
    return out


def stage_dedup(items: list[DataItem], recipe: Recipe) -> list[DataItem]:
    return dedup(items) if recipe.dedup else items


def stage_select(items: list[DataItem], recipe: Recipe) -> list[DataItem]:
    if recipe.curriculum:
        return curriculum_select(items, n_target=recipe.size)
    return items[: recipe.size] if recipe.size > 0 else items


def stage_emit(items: list[DataItem], recipe: Recipe) -> Manifest:
    emitter = build_emitter(recipe.emit)
    path = os.path.join(recipe.out_dir, f"{recipe.name}.{recipe.emit}.jsonl")
    return emitter.emit(items, path)


# -- orchestration -----------------------------------------------------------

def _items_through(stage: str, recipe: Recipe, config: Config) -> list[DataItem]:
    """Compute the item list up to and including `stage` (not emit)."""
    items = stage_load(recipe, config)
    if stage == "load":
        return items
    items = stage_generate(items, recipe, config)
    if stage == "generate":
        return items
    items = stage_dedup(items, recipe)
    if stage == "dedup":
        return items
    items = stage_select(items, recipe)
    return items  # "select"


def run_stage(stage: str, recipe: Recipe, config: Config | None = None):
    """Run up to one stage and return its output (item list, or Manifest for emit)."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}. Expected one of: {', '.join(STAGES)}.")
    config = config or Config.from_env()
    if stage == "emit":
        items = _items_through("select", recipe, config)
        return stage_emit(items, recipe)
    return _items_through(stage, recipe, config)


def run(recipe: Recipe, config: Config | None = None) -> RunResult:
    """Full pipeline: load → generate → dedup → select → emit → write manifest."""
    config = config or Config.from_env()
    items = _items_through("select", recipe, config)
    manifest = stage_emit(items, recipe)
    manifest.gaps_addressed = recipe.meta.get("gaps_addressed", [])
    report = build_report(recipe, items, manifest)
    report_path = write_report(report, recipe.out_dir, recipe.name)
    return RunResult(manifest=manifest, report_path=report_path, report=report)
