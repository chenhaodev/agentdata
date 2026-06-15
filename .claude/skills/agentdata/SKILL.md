---
name: agentdata
description: Turn a natural-language data request into a training dataset — diagnose a model/agent's weaknesses, pick the training regime, and build training-ready JSONL. Use when the user wants to prepare/select/generate training data, diagnose what to train, evaluate a memory system, or produce SFT/DPO/pretrain/chat/devset files. Trigger:/agentdata
trigger: /agentdata
---

# /agentdata — the second frontend of the one contract

This skill is **thin on purpose**. agentdata already has one execution path:
everything compiles to a `Recipe`, then `pipeline.run(recipe)` runs it. This skill
does **not** add a second code path — it translates the user's natural-language
request into the *same* `agentdata` CLI invocation and reports the result. If you
ever feel tempted to import internals and orchestrate stages by hand, stop: emit a
`Recipe` (flags or a YAML/JSON file) and shell out to `agentdata` instead.

Run from the repo root (or wherever `agentdata` is installed). If the console
script isn't on PATH, use `PYTHONPATH=src python3 -m agentdata …`.

## Decide which of the two verbs the request maps to

**diagnose** — the user has eval scores / a benchmark / a target to inspect, and
asks "what should I train?" / "where is it weak?" / "evaluate my memory system":

```bash
agentdata diagnose --report <scores.json>           # eval scores or a memory benchmark (agentmem's {hit1,hit3,mrr})
agentdata diagnose --scan  <path/to/SKILL.md|repo>  # no eval? statically scan for capability gaps
agentdata diagnose --report <scores.json> --out recipe.yaml   # also save the chosen Recipe
```
Report the gaps and the auto-selected recipe (regime / emit / sources / generate).

**build** — the user wants the actual data file. Prefer a saved recipe; otherwise
compile flags into one:

```bash
agentdata build --recipe recipe.yaml
agentdata build --source <spec[,spec...]> --emit <fmt> --size <N> --name <name>
```
- `--source`: `local:file.jsonl` · `hf:locomo` · `hf:jackrong-claude-opus-distill` ·
  `hf:agent-trajectory` · `hf:<id>` · `gh:owner/repo` · `kaggle:owner/dataset`.
- `--emit`: `sft` · `dpo` · `pretrain` · `chat` · `sharegpt` · `easydataset` · `devset`.
  (`dpo` synthesizes preference pairs on its own; `devset` is the few-shot eval set
  for prompt optimizers like GEPA/DSPy.)

Report the manifest line (count, path) and the audit manifest path.

## Pick the emit format from intent (don't ask if it's obvious)

- "fine-tune / instruction data / SFT" → `--emit sft` (or `chat` for multi-turn)
- "preference / DPO / alignment pairs" → `--emit dpo`
- "continue-pretrain / domain corpus" → `--emit pretrain`
- "for GEPA / DSPy / prompt optimization / a small eval set" → `--emit devset`
- "for LLaMA-Factory / easy-dataset" → `--emit easydataset`
- multi-agent / role-conditioned data → build with a recipe whose
  `generate: {recombine: true, multi_agent: true}` and `--emit chat`/`sharegpt`.

## Discover better data before committing (popularity matters)

If the user is unsure which dataset to use, rank the Hub by likes first:

```bash
agentdata sources --search "<topic>"     # likes/downloads = quality signal
agentdata sources                        # registry recipes + local files
```

## Guardrails

- **One contract**: only ever produce a `Recipe` and run the CLI. No second path.
- **Offline-first**: local/unify/diagnose/emit/select need no keys; only `hf:`/
  `kaggle:`/`physionet:` and real teacher generation touch the network/credentials.
- **Compliance**: PhysioNet/MIMIC items are non-redistributable and the emitters
  refuse to write them to shareable files — don't try to bypass that.
- Inspect one stage when debugging: `agentdata stage <load|generate|dedup|select|emit> --recipe …`.
