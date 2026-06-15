"""Benchmark the selection pipeline: dedup rate, curriculum quality, throughput.

Unlike a retrieval benchmark, the thing worth measuring here is whether
selection actually does its job: removes duplicates, biases toward harder /
more reasoning-dense samples, and orders them easy→hard — fast.

    python benchmark.py                              # uses ../dataset/sft_medical.jsonl if present
    python benchmark.py --source local:my.jsonl      # any source spec
    python benchmark.py --size 500

Metrics:
  dedup%        share of the pool removed as duplicates
  mean-diff     mean difficulty (pool → selected); hard-bias should raise it
  reasoning%    share of items carrying a <think> trace (pool → selected)
  monotonic     selection is ordered easy→hard (curriculum invariant)
  quartiles     selected count per difficulty quartile (target ~10/20/30/40%)
  throughput    items/sec for load and for dedup+select
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "src")

from agentdata.config import Config  # noqa: E402
from agentdata.emit.convert import to_messages  # noqa: E402
from agentdata.select import curriculum_select, dedup, score_difficulty  # noqa: E402
from agentdata.sources import load_sources  # noqa: E402
from agentdata.types import DataItem, KIND_MESSAGES  # noqa: E402


def _synthetic_pool(n: int = 4000) -> list[DataItem]:
    """Fallback pool when no local corpus is present: varied difficulty + 25% dupes."""
    items = []
    for i in range(n):
        cot = "<think>\n" + ("step " * (i % 120)) + "\n</think>\n" if i % 3 else ""
        items.append(DataItem(KIND_MESSAGES, messages=[
            {"role": "user", "content": f"question {i % (n * 3 // 4)} " + "context " * (i % 40)},
            {"role": "assistant", "content": cot + f"answer {i}"}],
            meta={"source": "synthetic"}))
    return items


def _reasoning_share(items: list[DataItem]) -> float:
    if not items:
        return 0.0
    n = sum(1 for it in items
            if any("<think>" in m["content"] for m in to_messages(it) if m["role"] == "assistant"))
    return n / len(items)


def _mean_diff(items: list[DataItem]) -> float:
    return sum(score_difficulty(it) for it in items) / len(items) if items else 0.0


def main(source: str, size: int) -> None:
    t0 = time.time()
    try:
        pool = load_sources([source], Config())
        if not pool:
            raise FileNotFoundError
    except Exception:
        print(f"(source {source!r} unavailable — using a synthetic pool)")
        pool = _synthetic_pool()
    load_s = time.time() - t0

    t1 = time.time()
    deduped = dedup(pool)
    selected = curriculum_select(deduped, n_target=size)
    sel_s = time.time() - t1

    import bisect

    dedup_pct = 100 * (len(pool) - len(deduped)) / len(pool) if pool else 0
    scores = [score_difficulty(it) for it in selected]
    monotonic = scores == sorted(scores)
    # bucket each selected item by its RANK in the sorted pool (how curriculum_select
    # actually splits quartiles) — value-thresholds would lump tied low scores into Q1.
    dd_scores = sorted(score_difficulty(it) for it in deduped)
    n = len(dd_scores)
    q = [0, 0, 0, 0]
    for s in scores:
        rank = bisect.bisect_left(dd_scores, s)
        q[min(rank * 4 // n, 3) if n else 0] += 1

    print("\n" + "=" * 70)
    print(f"{'metric':<14}{'pool':>14}{'selected':>14}")
    print("-" * 70)
    print(f"{'count':<14}{len(pool):>14}{len(selected):>14}")
    print(f"{'mean-diff':<14}{_mean_diff(deduped):>14.3f}{_mean_diff(selected):>14.3f}")
    print(f"{'reasoning%':<14}{100*_reasoning_share(deduped):>13.1f}%{100*_reasoning_share(selected):>13.1f}%")
    print("-" * 70)
    print(f"dedup%        {dedup_pct:.1f}%  ({len(pool)} → {len(deduped)})")
    print(f"monotonic     {'yes (easy→hard)' if monotonic else 'NO'}")
    print(f"quartiles     Q1={q[0]} Q2={q[1]} Q3={q[2]} Q4={q[3]}  (target ~10/20/30/40%)")
    print(f"throughput    load {len(pool)/load_s:,.0f}/s   dedup+select {len(deduped)/sel_s:,.0f}/s")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark agentdata selection quality.")
    ap.add_argument("--source", default="local:sft_medical.jsonl", help="source spec")
    ap.add_argument("--size", type=int, default=500, help="selection target")
    args = ap.parse_args()
    main(args.source, args.size)
