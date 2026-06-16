"""Closed-loop proof: agentdata-emitted DPO data → the real TRL DPOTrainer.

agentdata's `--emit dpo` writes {"prompt","chosen","rejected"} JSONL — byte-for-byte
TRL's DPO schema — so the file is fed to `trl.DPOTrainer` with zero reformatting.

This needs a trl/transformers pair where DPOTrainer imports (the `[train]` extra
pins one: trl==0.13.* with transformers>=4.46,<5). If that import fails in your
env, use `dpo_min.py`, which runs the same DPO objective with no trl dependency on
the identical file. A from-config tiny model keeps this offline and CPU-only.

    pip install -e '.[train]'
    python train/dpo_trl.py
    python train/dpo_trl.py --steps 60
"""

from __future__ import annotations

import argparse
import os
import tempfile

from train._common import emit, loss_drop, no_wandb, tiny_gpt2
from train.tiny_tokenizer import build_tiny_tokenizer


def train(seed: str, steps: int, size: int) -> tuple[float, float]:
    no_wandb()
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    with tempfile.TemporaryDirectory() as work:
        dpo_path = emit(seed, work, "dpo", "dpo", size, "dpo_pairs")
        ds = load_dataset("json", data_files=dpo_path, split="train")
        print(f"[trl] loaded {len(ds)} pairs; columns={ds.column_names}")

        tok = build_tiny_tokenizer()
        model = tiny_gpt2(tok.vocab_size)

        args = DPOConfig(
            output_dir=os.path.join(work, "ckpt"),
            max_steps=steps, per_device_train_batch_size=2,
            learning_rate=5e-3, logging_steps=1, report_to=[],
            beta=0.1, max_length=128, max_prompt_length=64,
            save_strategy="no", seed=0,
        )
        trainer = DPOTrainer(
            model=model, ref_model=None, args=args,
            train_dataset=ds, processing_class=tok,
        )
        trainer.train()
        return loss_drop(trainer.state.log_history)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(__file__)
    p.add_argument("--seed", default=os.path.join(here, "fixtures", "chat_seed.jsonl"))
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--size", type=int, default=0, help="cap rows (0 = all)")
    args = p.parse_args(argv)

    first, last = train(args.seed, args.steps, args.size)
    drop = (first - last) / first * 100 if first else 0.0
    print(f"\n[result] DPO loss {first:.3f} -> {last:.3f}  ({drop:+.1f}% lower)")
    ok = last < first
    print("[result] downstream DPO loop (real trl.DPOTrainer):", "PASS ✅" if ok else "NO LEARNING ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
