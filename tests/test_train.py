"""Downstream-training proof. Run: python tests/test_train.py

Two layers:
  * always-on (offline, agentdata only): the emitted files match the *exact*
    schemas TRL's trainers consume — chat `{"messages":[{"role","content"}]}` for
    SFTTrainer, `{"prompt","chosen","rejected"}` for DPOTrainer. This is the
    format contract; it runs in CI with core deps only.
  * opt-in (RUN_TRAIN=1, needs torch+trl): actually trains a tiny model on those
    files via `train/sft_trl.py` (real trl.SFTTrainer) and `train/dpo_min.py`
    (standard DPO objective) and asserts the loss falls — the loop closes.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentdata import Config, DatasetBuilder, Recipe  # noqa: E402

TRAIN = os.getenv("RUN_TRAIN") == "1"
E2E = os.getenv("RUN_E2E") == "1"  # real Hub data + real pretrained model (network + minutes)
_SEED = os.path.join(os.path.dirname(__file__), "..", "train", "fixtures", "chat_seed.jsonl")


def _emit(fmt: str, regime: str) -> list[dict]:
    seed = os.path.abspath(_SEED)
    with tempfile.TemporaryDirectory() as work:
        cfg = Config(data_dir=os.path.dirname(seed), out_dir=work)
        recipe = Recipe(sources=[f"local:{os.path.basename(seed)}"],
                        emit=fmt, regime=regime, name="t", out_dir=work)
        result = DatasetBuilder(cfg).build(recipe)
        with open(result.manifest.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


def test_chat_matches_sfttrainer_schema():
    rows = _emit("chat", "sft")
    assert rows, "chat emit produced nothing"
    for r in rows:
        assert set(r) == {"messages"}, f"unexpected keys: {set(r)}"
        assert len(r["messages"]) >= 2
        for m in r["messages"]:
            assert set(m) >= {"role", "content"}
            assert m["role"] in {"system", "user", "assistant", "tool"}
            assert m["content"].strip()
    print(f"ok  chat schema matches trl.SFTTrainer ({len(rows)} rows)")


def test_dpo_matches_dpotrainer_schema():
    rows = _emit("dpo", "dpo")
    assert rows, "dpo emit produced nothing"
    for r in rows:
        assert set(r) == {"prompt", "chosen", "rejected"}, f"unexpected keys: {set(r)}"
        assert r["prompt"].strip() and r["chosen"].strip() and r["rejected"].strip()
        assert r["chosen"].strip() != r["rejected"].strip()
    print(f"ok  dpo schema matches trl.DPOTrainer ({len(rows)} pairs)")


def test_sft_loop_reduces_loss():
    if not TRAIN:
        print("skip test_sft_loop_reduces_loss (RUN_TRAIN != 1)")
        return
    from train.sft_trl import train

    first, last = train(os.path.abspath(_SEED), steps=30, size=0)
    assert last < first, f"SFT loss did not fall: {first:.3f} -> {last:.3f}"
    print(f"ok  SFT loop reduces loss {first:.3f} -> {last:.3f}")


def test_dpo_loop_reduces_loss():
    if not TRAIN:
        print("skip test_dpo_loop_reduces_loss (RUN_TRAIN != 1)")
        return
    # prefer the real trl.DPOTrainer (needs the pinned [train] extra); fall back to
    # the trl-free DPO objective when DPOTrainer can't import (version skew).
    from train._common import no_wandb

    no_wandb()
    try:
        from trl import DPOTrainer  # noqa: F401

        from train.dpo_trl import train

        which = "trl.DPOTrainer"
    except Exception:
        from train.dpo_min import train

        which = "dpo_min (trl-free)"

    first, last = train(os.path.abspath(_SEED), steps=30, size=0)
    assert last < first, f"DPO loss did not fall: {first:.3f} -> {last:.3f}"
    print(f"ok  DPO loop reduces loss {first:.3f} -> {last:.3f}  [{which}]")


def test_e2e_real_holdout_improves():
    # the strongest proof: real Hub data + real pretrained model, held-out loss falls.
    # opt-in (RUN_E2E=1) — pulls a dataset and a model and trains for a few minutes.
    if not E2E:
        print("skip test_e2e_real_holdout_improves (RUN_E2E != 1)")
        return
    from train.e2e_real import train

    endpoint = os.getenv("HF_ENDPOINT_E2E")  # e.g. https://huggingface.co if the mirror is down
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    before, after = train("hf:alpaca-cleaned", "distilgpt2", size=200, steps=40)
    assert after < before, f"held-out eval loss did not fall: {before:.3f} -> {after:.3f}"
    print(f"ok  e2e real held-out eval loss {before:.3f} -> {after:.3f}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} ran")


if __name__ == "__main__":
    main()
