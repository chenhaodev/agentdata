"""agentdata — data-select + generator middleware between raw sources and trainers.

Pull from HuggingFace / Kaggle / PhysioNet / GitHub / local, unify to one shape,
diagnose a target's weaknesses to auto-pick a training regime + dataset types,
optionally synthesize/recombine data, then emit training-ready JSONL that
llm-trainer / easy-dataset / LLaMA-Factory consume directly.

    from agentdata import DatasetBuilder, Recipe

    recipe = Recipe(sources=["local"], emit="sft", size=200)
    manifest = DatasetBuilder().build(recipe)   # writes out/dataset.sft.jsonl
    print(manifest.count, "samples")

One contract, two frontends: CLI and skill both compile to a Recipe, then call
pipeline.run(recipe) — never two code paths.
"""

from .builder import DatasetBuilder
from .config import Config
from .diagnose import Diagnoser
from .types import DataItem, Diagnosis, Manifest, Recipe

__all__ = [
    "DatasetBuilder",
    "Diagnoser",
    "Config",
    "DataItem",
    "Recipe",
    "Diagnosis",
    "Manifest",
]
__version__ = "0.1.0"
