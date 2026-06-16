"""End-to-end on real data + a real pretrained model: held-out loss must fall.

The strongest proof in this repo. Everything is real except the scale:
  real Hub dataset (hf:alpaca-cleaned) -> agentdata (unify → curriculum select →
  emit chat) -> real `trl.SFTTrainer` on a real pretrained model (distilgpt2) ->
  the **held-out** eval loss drops. Held-out = the model improves on examples it
  never trained on, i.e. it generalized, not memorized.

Needs network (dataset via agentdata's resolve-URL fetch; model via the Hub) and
a few minutes of CPU. Opt-in only — not in CI. If your `HF_ENDPOINT` points at a
mirror that's down, pass `--hf-endpoint https://huggingface.co`.

    pip install -e '.[train,hf]'
    python train/e2e_real.py
    python train/e2e_real.py --model distilgpt2 --source hf:alpaca-cleaned --size 400 --steps 60
"""

from __future__ import annotations

import argparse
import os

from train._common import no_wandb

# distilgpt2's tiny chat rendering — role-prefixed turns + an EOS terminator.
_CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}{{ eos_token }}"
)


def emit_chat(source: str, out_dir: str, size: int) -> str:
    """Real Hub source → agentdata pipeline → chat JSONL; return the path."""
    from agentdata.builder import DatasetBuilder
    from agentdata.config import Config
    from agentdata.types import Recipe

    cfg = Config(cache_dir=".data/cache", out_dir=out_dir)
    recipe = Recipe(sources=[source], emit="chat", regime="sft",
                    size=size, name="e2e_real", out_dir=out_dir)
    result = DatasetBuilder(cfg).build(recipe)
    print(f"[agentdata] {source} -> {result.manifest.count} chat rows ({result.manifest.path})")
    return result.manifest.path


def train(source: str, model_id: str, size: int, steps: int,
          test_frac: float = 0.2) -> tuple[float, float]:
    no_wandb()
    import tempfile

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    with tempfile.TemporaryDirectory() as work:
        chat_path = emit_chat(source, work, size)
        split = load_dataset("json", data_files=chat_path, split="train").train_test_split(
            test_size=test_frac, seed=0)
        print(f"[trl] train={len(split['train'])} eval(held-out)={len(split['test'])}")

        tok = AutoTokenizer.from_pretrained(model_id)
        tok.pad_token = tok.pad_token or tok.eos_token
        tok.chat_template = _CHAT_TEMPLATE
        model = AutoModelForCausalLM.from_pretrained(model_id)

        def render(batch):
            return tok.apply_chat_template(batch["messages"], tokenize=False)

        args = SFTConfig(
            output_dir=os.path.join(work, "ckpt"),
            max_steps=steps, per_device_train_batch_size=4,
            learning_rate=2e-4, logging_steps=max(steps // 3, 1), report_to=[],
            max_seq_length=256, save_strategy="no", seed=0, eval_strategy="no",
        )
        trainer = SFTTrainer(
            model=model, args=args, train_dataset=split["train"],
            eval_dataset=split["test"], processing_class=tok, formatting_func=render,
        )
        before = trainer.evaluate()["eval_loss"]
        trainer.train()
        after = trainer.evaluate()["eval_loss"]
        return before, after


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="hf:alpaca-cleaned", help="agentdata source spec")
    p.add_argument("--model", default="distilgpt2", help="real pretrained base model")
    p.add_argument("--size", type=int, default=400, help="examples to pull + split")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--hf-endpoint", default="", help="override HF_ENDPOINT for model fetch")
    args = p.parse_args(argv)

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    before, after = train(args.source, args.model, args.size, args.steps)
    drop = (before - after) / before * 100 if before else 0.0
    print(f"\n[result] held-out eval loss {before:.4f} -> {after:.4f}  ({drop:+.1f}% lower)")
    ok = after < before
    print("[result] real-data + real-model loop:", "PASS ✅ (generalized)" if ok else "NO GAIN ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
