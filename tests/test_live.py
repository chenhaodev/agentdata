"""Opt-in live tests. Skipped unless RUN_LIVE=1 (so plain `pytest` stays offline).

Exercises the real network paths: a tiny HuggingFace split → unify → emit. Kaggle
and PhysioNet are gated behind creds and self-skip when absent.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

LIVE = os.getenv("RUN_LIVE") == "1"


def _skip(reason: str) -> bool:
    print(f"skip {reason}")
    return True


def test_live_hf_roundtrip():
    if not LIVE:
        return _skip("RUN_LIVE != 1")
    try:
        import datasets  # noqa: F401
    except ImportError:
        return _skip("datasets not installed (pip install 'agentdata[hf]')")

    from agentdata import Config, DatasetBuilder, Recipe

    # a small, public dataset id; override with AGENTDATA_LIVE_HF if needed
    spec = os.getenv("AGENTDATA_LIVE_HF", "hf:tatsu-lab/alpaca:train[:50]")
    with tempfile.TemporaryDirectory() as d:
        cfg = Config(out_dir=os.path.join(d, "out"))
        result = DatasetBuilder(cfg).build(
            Recipe(sources=[spec], emit="sft", size=10, name="live_hf")
        )
        assert result.manifest.count > 0, "live HF build produced nothing"
        assert os.path.exists(result.manifest.path)
    print(f"ok  live HF roundtrip: {result.manifest.count} samples from {spec}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} ran")


if __name__ == "__main__":
    main()
