"""Shared helpers for the downstream-training proofs.

Keeps the SFT and DPO scripts to just their trainer wiring. Everything here is
offline: agentdata emits the data, a from-config GPT-2 is the student, and a
broken/optional wandb is neutralized so a real trainer runs with no tracker.
"""

from __future__ import annotations

import os
import sys

# make `src/` and the repo root importable when run from the repo without an install
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def no_wandb() -> None:
    """Make trl/transformers treat wandb as absent (a half-installed one crashes import)."""
    import importlib.machinery
    import types

    stub = types.ModuleType("wandb")  # no `run` attr ⇒ transformers treats it as absent
    stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)  # accelerate find_spec
    sys.modules["wandb"] = stub


def emit(seed_path: str, out_dir: str, fmt: str, regime: str, size: int, name: str) -> str:
    """Run agentdata to turn a raw seed file into a training-ready JSONL; return its path."""
    from agentdata.builder import DatasetBuilder
    from agentdata.config import Config
    from agentdata.types import Recipe

    data_dir = os.path.dirname(os.path.abspath(seed_path))
    cfg = Config(data_dir=data_dir, out_dir=out_dir)
    recipe = Recipe(
        sources=[f"local:{os.path.basename(seed_path)}"], emit=fmt, regime=regime,
        size=size, name=name, out_dir=out_dir,
    )
    result = DatasetBuilder(cfg).build(recipe)
    print(f"[agentdata] emitted {result.manifest.count} {fmt} rows -> {result.manifest.path}")
    return result.manifest.path


def tiny_gpt2(vocab_size: int):
    """A 2-layer GPT-2 built from config — no download, no pretrained weights."""
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=vocab_size, n_positions=256, n_embd=64,
        n_layer=2, n_head=2, bos_token_id=1, eos_token_id=2,
    )
    return GPT2LMHeadModel(cfg)


def loss_drop(log_history: list[dict]) -> tuple[float, float]:
    """First and last training loss from a Trainer's log history."""
    losses = [r["loss"] for r in log_history if "loss" in r]
    if not losses:
        raise RuntimeError("no training loss was logged")
    return losses[0], losses[-1]
