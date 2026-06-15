"""agentdata demo — diagnose → recipe → build, fully offline.

Run: python demo.py
Uses an in-memory eval report and a tiny in-memory local corpus so it needs no
network, no keys, and no files beyond what it writes to ./out.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agentdata import Config, DatasetBuilder, Diagnoser, Recipe  # noqa: E402


def main() -> None:
    # 1) diagnose a target's eval scores -> an auto-selected Recipe
    report = {"scores": {"single_hop": 0.8, "multi_hop": 0.4, "temporal": 0.35,
                         "math": 0.3, "reasoning": 0.5}}
    dgn = Diagnoser()
    dx, recipe = dgn.diagnose(report=report, name="demo")
    print("gaps:", dx.gaps)
    print("auto recipe:", f"regime={recipe.regime} emit={recipe.emit} "
          f"sources={recipe.sources} generate={recipe.generate}\n")

    # 2) build a real dataset from a tiny local corpus (offline)
    with tempfile.TemporaryDirectory() as d:
        corpus = os.path.join(d, "mini.jsonl")
        with open(corpus, "w", encoding="utf-8") as f:
            f.write('{"instruction": "What is hypertension?", "input": "", '
                    '"output": "High blood pressure, a chronic condition."}\n')
            f.write('{"conversations": [{"from": "human", "value": "Define CKD."}, '
                    '{"from": "gpt", "value": "Chronic kidney disease."}]}\n')
        cfg = Config(data_dir=d, out_dir=os.path.join(d, "out"))
        build_recipe = Recipe(sources=[f"local:{corpus}"], emit="sft", name="demo")
        result = DatasetBuilder(cfg).build(build_recipe)
        print(f"built {result.manifest.count} SFT samples -> {result.manifest.path}")
        print(f"manifest -> {result.report_path}")
        with open(result.manifest.path, encoding="utf-8") as f:
            print("first row:", f.readline().strip())


if __name__ == "__main__":
    main()
