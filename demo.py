"""agentdata demo — diagnose → recipe → build, plus the memory-eval and
multi-agent paths. Fully offline.

Run: python demo.py
Uses in-memory reports/corpora so it needs no network, no keys, and no files
beyond what it writes to a temp dir.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agentdata import Config, DatasetBuilder, Diagnoser, Recipe  # noqa: E402
from agentdata.emit import build_emitter  # noqa: E402
from agentdata.emit.convert import to_messages  # noqa: E402
from agentdata.generate import recombine  # noqa: E402
from agentdata.types import KIND_MESSAGES, DataItem  # noqa: E402


def _h(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def demo_diagnose_and_build() -> None:
    """1) eval scores → auto recipe; 2) build a real dataset from a tiny corpus."""
    _h("1) diagnose eval scores → auto recipe → build")
    report = {"scores": {"single_hop": 0.8, "multi_hop": 0.4, "temporal": 0.35,
                         "math": 0.3, "reasoning": 0.5}}
    dx, recipe = Diagnoser().diagnose(report=report, name="demo")
    print("gaps:", dx.gaps)
    print(f"auto recipe: regime={recipe.regime} emit={recipe.emit} "
          f"sources={recipe.sources} generate={recipe.generate}")

    with tempfile.TemporaryDirectory() as d:
        corpus = os.path.join(d, "mini.jsonl")
        with open(corpus, "w", encoding="utf-8") as f:
            f.write('{"instruction": "What is hypertension?", "input": "", '
                    '"output": "High blood pressure, a chronic condition."}\n')
            f.write('{"conversations": [{"from": "human", "value": "Define CKD."}, '
                    '{"from": "gpt", "value": "Chronic kidney disease."}]}\n')
        out = os.path.join(d, "out")
        result = DatasetBuilder(Config(data_dir=d)).build(
            Recipe(sources=[f"local:{corpus}"], emit="sft", name="demo", out_dir=out))
        print(f"built {result.manifest.count} SFT samples -> {result.manifest.path}")
        with open(result.manifest.path, encoding="utf-8") as f:
            print("first row:", f.readline().strip())


def demo_evaluate_memory() -> None:
    """Diagnose a memory system from its benchmark JSON (agentmem's format):
    weakest backend chosen, timings ignored → a LoCoMo-recombined recipe."""
    _h("2) evaluate a memory system (agentmem benchmark → memory recipe)")
    bench = [
        {"backend": "vector", "ok": True, "stored": 12, "write_s": 0.3, "query_ms": 4.1,
         "hit1": 0.42, "hit3": 0.55, "clean": 0.38, "mrr": 0.49},
        {"backend": "mem0", "ok": True, "stored": 12, "write_s": 6.2, "query_ms": 88.0,
         "hit1": 0.58, "hit3": 0.71, "clean": 0.50, "mrr": 0.63},
    ]
    dx, recipe = Diagnoser().diagnose(report=bench, name="memfix")
    print("memory gaps (weakest backend = vector):", dx.gaps)
    print(f"  memory capability = {dx.scores['memory']:.2f}  (hit1={dx.scores['hit1']})")
    print(f"recipe: regime={recipe.regime} emit={recipe.emit} "
          f"sources={recipe.sources} generate={recipe.generate}")
    print("  → 'weak retrieval → recombine LoCoMo long conversations to fill the gap'")


def demo_multi_agent() -> None:
    """Recombine similar subjects into one role-conditioned multi-agent transcript;
    chat/sharegpt emitters preserve the agent roles."""
    _h("3) generate multi-agent training data (role-conditioned recombination)")
    subjects = [
        DataItem(KIND_MESSAGES, meta={"source": "demo", "tags": ["trip"]}, messages=[
            {"role": "user", "content": "plan a 3-day trip together"},
            {"role": "assistant", "content": "Day 1 old town, Day 2 coast, Day 3 hills"}]),
        DataItem(KIND_MESSAGES, meta={"source": "demo", "tags": ["trip"]}, messages=[
            {"role": "user", "content": "plan a 3-day trip together"},
            {"role": "assistant", "content": "Add a budget cap and book trains early"}]),
    ]
    ma = recombine(subjects, multi_agent=True)[0]
    print(f"recombined {ma.meta['n_subjects']} subjects into a multi-agent transcript:")
    for m in to_messages(ma):
        print(f"  {m['role']:>8}: {m['content']}")

    # prove the agent roles survive all the way to the emitted file
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "multiagent.sharegpt.jsonl")
        build_emitter("sharegpt").emit([ma], path)
        row = json.loads(open(path, encoding="utf-8").readline())
        print("roles preserved in sharegpt output:", [t["from"] for t in row["conversations"]])


def main() -> None:
    demo_diagnose_and_build()
    demo_evaluate_memory()
    demo_multi_agent()
    print()


if __name__ == "__main__":
    main()
