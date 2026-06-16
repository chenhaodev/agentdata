"""Closed-loop proof: agentdata-emitted DPO data → a preference-optimization step.

agentdata's `--emit dpo` writes {"prompt","chosen","rejected"} JSONL — byte-for-byte
TRL's DPOTrainer schema (see the assertion in tests/test_train.py). The companion
`sft_trl.py` runs the *real* `trl.SFTTrainer`; TRL's DPOTrainer, however, does not
import under the installed transformers (a version skew that's the environment's,
not ours), so this file runs a compact but standard DPO objective directly on the
same file to prove the pairs actually drive a preference loss down:

    L = -log σ( β · [ (logπ_chosen - logπ_rejected) - (logπ_ref_chosen - logπ_ref_rejected) ] )

where the reference is a frozen copy of the initial policy. Swap `tiny_gpt2()` for a
`from_pretrained` model and this is a real (if minimal) DPO fine-tune.

    python train/dpo_min.py
    python train/dpo_min.py --steps 60
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile

from train._common import emit, tiny_gpt2
from train.tiny_tokenizer import build_tiny_tokenizer


def _seq_logp(model, tok, prompts, completions, device):
    """Sum log-prob of each completion's tokens given its prompt (prompt masked)."""
    import torch
    import torch.nn.functional as F

    totals = []
    for prompt, completion in zip(prompts, completions):
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = torch.tensor([p_ids + c_ids], device=device)
        logits = model(ids).logits[0, :-1]  # predict token t+1 from t
        logp = F.log_softmax(logits, dim=-1)
        target = ids[0, 1:]
        tok_logp = logp[range(len(target)), target]
        comp_logp = tok_logp[max(len(p_ids) - 1, 0):]  # only completion positions
        totals.append(comp_logp.sum())
    return torch.stack(totals)


def train(seed: str, steps: int, size: int, beta: float = 0.1) -> tuple[float, float]:
    import torch

    with tempfile.TemporaryDirectory() as work:
        dpo_path = emit(seed, work, "dpo", "dpo", size, "dpo_pairs")
        with open(dpo_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        print(f"[dpo] loaded {len(rows)} preference pairs from {os.path.basename(dpo_path)}")

    device = "cpu"
    tok = build_tiny_tokenizer()
    policy = tiny_gpt2(tok.vocab_size).to(device).train()
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=5e-3)
    prompts = [r["prompt"] for r in rows]
    chosen = [r["chosen"] for r in rows]
    rejected = [r["rejected"] for r in rows]

    first = last = 0.0
    for step in range(steps):
        pol_c = _seq_logp(policy, tok, prompts, chosen, device)
        pol_r = _seq_logp(policy, tok, prompts, rejected, device)
        with torch.no_grad():
            ref_c = _seq_logp(ref, tok, prompts, chosen, device)
            ref_r = _seq_logp(ref, tok, prompts, rejected, device)
        margin = (pol_c - pol_r) - (ref_c - ref_r)
        loss = -torch.nn.functional.logsigmoid(beta * margin).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        val = loss.item()
        if step == 0:
            first = val
        last = val
        if step % 5 == 0 or step == steps - 1:
            acc = float((margin > 0).float().mean())
            print(f"  step {step:3d}  loss {val:.4f}  pref-acc {acc:.2f}")
    return first, last


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
    print("[result] downstream DPO loop:", "PASS ✅" if ok else "NO LEARNING ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
