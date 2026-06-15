"""DatasetBuilder — the public facade over the pipeline.

Compiles a Recipe (object, dict, or YAML/JSON file) and runs it. Both the CLI and
any future skill go through here, honoring the one-contract rule.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .config import Config
from .pipeline import RunResult, run, run_stage
from .types import Recipe


def load_recipe(src: "str | dict[str, Any] | Recipe") -> Recipe:
    """Coerce a recipe object / dict / path (.yaml|.yml|.json) into a Recipe."""
    if isinstance(src, Recipe):
        return src
    if isinstance(src, dict):
        return Recipe.from_dict(src)
    if isinstance(src, str):
        ext = os.path.splitext(src)[1].lower()
        with open(src, encoding="utf-8") as f:
            if ext in (".yaml", ".yml"):
                import yaml  # pyyaml is a core dep

                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return Recipe.from_dict(data)
    raise TypeError(f"Cannot load recipe from {type(src).__name__}")


class DatasetBuilder:
    def __init__(self, config: Config | None = None):
        self.config = config or Config.from_env()

    def build(self, recipe: "str | dict[str, Any] | Recipe") -> RunResult:
        """Run the full pipeline for a recipe and return the result + manifest."""
        return run(load_recipe(recipe), self.config)

    def stage(self, stage: str, recipe: "str | dict[str, Any] | Recipe"):
        """Run a single pipeline stage (for inspection / debugging)."""
        return run_stage(stage, load_recipe(recipe), self.config)
