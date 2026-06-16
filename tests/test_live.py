"""Opt-in live tests. Skipped unless RUN_LIVE=1 (so plain `pytest` stays offline).

Exercises the real network paths: the named HuggingFace registry recipes
(`hf:locomo`, `hf:jackrong-claude-opus-distill`) → unify → emit. The registry ids
+ data files were validated against the Hub (2026-06-15 / 2026-06-16):
  - Percena/locomo-mc10 :: data/locomo_mc10.json  → 1986 QA items (question_type tags)
  - Jackrong/Claude-opus-4.6-TraceInversion-9000x  → 8669 message items (<think> traces)
  - yahma/alpaca-cleaned :: alpaca_data_cleaned.json  → 51,760 instruction items
  - NousResearch/hermes-function-calling-v1 :: func-calling-singleturn.json  → 1893 items
Downloads can be slow when the Hub's Xet/CDN backend is unreachable (the source
falls back to a direct resolve-URL fetch); these are opt-in and not run in CI.
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


def _have_hf() -> bool:
    try:
        import datasets  # noqa: F401
        import huggingface_hub  # noqa: F401

        return True
    except ImportError:
        return False


def test_live_locomo_registry():
    """LoCoMo recipe loads as QA items carrying per-row question_type tags."""
    if not LIVE:
        return _skip("RUN_LIVE != 1")
    if not _have_hf():
        return _skip("datasets/huggingface_hub not installed (pip install 'agentdata[hf]')")

    from agentdata.config import Config
    from agentdata.sources import build_source

    items = build_source("hf", Config(cache_dir=".data/cache")).load("locomo")
    assert items, "locomo returned no items"
    assert all(it.kind == "qa" for it in items[:20])
    # the LoCoMo categories are exactly the diagnosis axes — they must reach meta.tags
    cats = {t for it in items for t in it.meta.get("tags", [])}
    assert cats & {"single_hop", "multi_hop", "temporal", "open_domain"}, cats
    print(f"ok  live locomo: {len(items)} QA items, categories tagged")


def test_live_jackrong_registry_has_reasoning():
    """The distilled reasoning recipe loads as chat items with <think> traces."""
    if not LIVE:
        return _skip("RUN_LIVE != 1")
    if not _have_hf():
        return _skip("datasets/huggingface_hub not installed")

    from agentdata.config import Config
    from agentdata.emit.convert import to_messages
    from agentdata.sources import build_source

    items = build_source("hf", Config(cache_dir=".data/cache")).load("jackrong-claude-opus-distill")
    assert items, "jackrong returned no items"
    with_think = sum(1 for it in items[:200]
                     if any("<think>" in m["content"]
                            for m in to_messages(it) if m["role"] == "assistant"))
    assert with_think > 0, "no <think> reasoning traces found in distilled set"
    print(f"ok  live jackrong: {len(items)} items, {with_think}/200 carry <think> traces")


def test_live_hf_build_roundtrip():
    """End-to-end: registry recipe → unify → SFT emit → non-empty manifest."""
    if not LIVE:
        return _skip("RUN_LIVE != 1")
    if not _have_hf():
        return _skip("datasets/huggingface_hub not installed")

    from agentdata import Config, DatasetBuilder, Recipe

    with tempfile.TemporaryDirectory() as d:
        cfg = Config(cache_dir=".data/cache", out_dir=os.path.join(d, "out"))
        result = DatasetBuilder(cfg).build(
            Recipe(sources=["hf:locomo"], emit="sft", size=50, name="live_locomo")
        )
        assert result.manifest.count > 0, "live HF build produced nothing"
        assert os.path.exists(result.manifest.path)
        assert result.report["provenance"]["by_source"].get("huggingface", 0) > 0
    print(f"ok  live HF build: {result.manifest.count} SFT samples from hf:locomo")


def test_live_alpaca_cleaned_registry():
    """The cleaned-Alpaca recipe loads as instruction (alpaca) chat items."""
    if not LIVE:
        return _skip("RUN_LIVE != 1")
    if not _have_hf():
        return _skip("datasets/huggingface_hub not installed")

    from agentdata.config import Config
    from agentdata.sources import build_source

    items = build_source("hf", Config(cache_dir=".data/cache")).load("alpaca-cleaned")
    assert items, "alpaca-cleaned returned no items"
    assert all(it.kind == "messages" for it in items[:20])
    assert all(it.meta.get("format") == "alpaca" for it in items[:20])
    print(f"ok  live alpaca-cleaned: {len(items)} instruction items")


def test_live_hermes_function_calling_registry():
    """The Hermes function-calling recipe loads as sharegpt items with tool/system
    roles and per-row category tags."""
    if not LIVE:
        return _skip("RUN_LIVE != 1")
    if not _have_hf():
        return _skip("datasets/huggingface_hub not installed")

    from agentdata.config import Config
    from agentdata.emit.convert import to_messages
    from agentdata.sources import build_source

    items = build_source("hf", Config(cache_dir=".data/cache")).load("hermes-function-calling")
    assert items, "hermes-function-calling returned no items"
    roles = {m["role"] for it in items[:50] for m in to_messages(it)}
    assert {"system", "assistant"} <= roles, roles
    cats = {t for it in items for t in it.meta.get("tags", [])}
    assert cats - {"agent", "tool-use", "function-calling"}, "no per-row category tags"
    print(f"ok  live hermes-function-calling: {len(items)} items, roles={sorted(roles)}")


def test_live_github_source():
    """GitHub source against a public repo (no creds): code/docs → KIND_TEXT items."""
    if not LIVE:
        return _skip("RUN_LIVE != 1")
    try:
        import requests  # noqa: F401
    except ImportError:
        return _skip("requests not installed (pip install 'agentdata[github]')")

    from agentdata.config import Config
    from agentdata.sources import build_source

    repo = os.getenv("AGENTDATA_LIVE_GH", "chenhaodev/agentdata")
    items = build_source("gh", Config(cache_dir=".data/cache")).load(repo)
    assert items, f"github source returned nothing for {repo}"
    assert all(it.kind == "text" and it.meta.get("source") == "github" for it in items)
    assert any(it.meta.get("path", "").endswith(".py") for it in items)
    print(f"ok  live github: {len(items)} text items from {repo}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} ran")


if __name__ == "__main__":
    main()
