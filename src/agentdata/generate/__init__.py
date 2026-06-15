"""Synthetic data generation: teacher synth, subject recombination, GEPA traces."""

from .gepa import attach_feedback, keep_high_signal
from .llm import LLMProvider, get_provider
from .preference import attach_rejected
from .recombine import recombine
from .synth import synth

__all__ = [
    "synth",
    "recombine",
    "attach_rejected",
    "attach_feedback",
    "keep_high_signal",
    "get_provider",
    "LLMProvider",
]
