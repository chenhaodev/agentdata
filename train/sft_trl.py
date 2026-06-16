"""Closed-loop proof: agentdata-emitted SFT data → a real TRL SFTTrainer.

This is the consumer-side test of the whole package. It does NOT reformat the
file: agentdata emits chat JSONL, and `trl.SFTTrainer` ingests that exact file
(the chat template is applied at train time, as in any real SFT pipeline). A
from-config tiny model + offline tokenizer keep it runnable with no download and
no GPU; the only claim is "a standard trainer accepts our output and its loss
falls", which is what proves the emitted format actually trains a model.

To train a real model, swap `build_tiny_tokenizer()`/`tiny_gpt2()` for
`AutoTokenizer`/`AutoModelForCausalLM.from_pretrained(BASE)` and point `--seed`
at any agentdata `--emit chat` output.

    python train/sft_trl.py                 # fixture -> agentdata -> SFTTrainer
    python train/sft_trl.py --steps 60
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
    from trl import SFTConfig, SFTTrainer

    with tempfile.TemporaryDirectory() as work:
        chat_path = emit(seed, work, "chat", "sft", size, "sft_chat")
        ds = load_dataset("json", data_files=chat_path, split="train")
        print(f"[trl] loaded {len(ds)} rows; columns={ds.column_names}")

        tok = build_tiny_tokenizer()
        model = tiny_gpt2(tok.vocab_size)

        args = SFTConfig(
            output_dir=os.path.join(work, "ckpt"),
            max_steps=steps, per_device_train_batch_size=2,
            learning_rate=5e-3, logging_steps=1, report_to=[],
            max_seq_length=128, save_strategy="no", seed=0,
        )

        # standard SFT step: render each {"messages":[...]} row through the chat
        # template at train time (the file on disk stays agentdata's chat JSONL).
        def render(batch):
            return tok.apply_chat_template(batch["messages"], tokenize=False)

        trainer = SFTTrainer(
            model=model, args=args, train_dataset=ds, processing_class=tok,
            formatting_func=render,
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
    print(f"\n[result] SFT loss {first:.3f} -> {last:.3f}  ({drop:+.1f}% lower)")
    ok = last < first
    print("[result] downstream SFT loop:", "PASS ✅" if ok else "NO LEARNING ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
