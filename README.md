# agentdata

**Data-select + generator middleware that sits between raw data sources and training pipelines.**

Pull from HuggingFace / Kaggle / PhysioNet / GitHub / local files, **unify the
format**, **diagnose a target agent/model's weaknesses** and auto-pick the
training regime (pretrain / continue-pretrain / SFT / DPO / GRPO) + dataset types,
optionally **synthesize** data (teacher generation, subject recombination, GEPA
feedback traces), then **emit training-ready JSONL** that `llm-trainer`,
`easy-dataset`, and LLaMA-Factory consume directly — with a provenance/audit
manifest.

```python
from agentdata import DatasetBuilder, Diagnoser, Recipe

# 1) diagnose → auto recipe
dx, recipe = Diagnoser().diagnose(report={"scores": {"temporal": 0.35, "math": 0.3}})
print(recipe.regime, recipe.sources)        # grpo  ['hf:jackrong-...', 'hf:locomo']

# 2) build a dataset from a recipe (offline, from local corpora)
result = DatasetBuilder().build(Recipe(sources=["local:sft_medical.jsonl"], emit="sft", size=200))
print(result.manifest.count, "→", result.manifest.path)
```

It mirrors the [`agentmem`](../agentmem) architecture — Protocol-based pluggable
backends + factory + facade + env config + offline-first tests — so it's a clean
sibling, not another one-off skill.

## Why

The pieces existed but were scattered and domain-locked (synthetic generation,
dataset selection/curriculum, multi-backend trainers, pre-built corpora).
`agentdata` is the general, domain-agnostic middleware tying them together behind
**one contract**: `agentdata diagnose <target>` → `agentdata build <recipe>`
produces a versioned, validated dataset in the exact format the chosen trainer
expects.

## Install

```bash
pip install -e .                 # core: local sources, unify, emit, select, diagnose
pip install -e '.[hf]'           # + HuggingFace source
pip install -e '.[gen]'          # + synthetic generation (Anthropic teacher)
pip install -e '.[all,dev]'      # everything + pytest
```

Nothing above core is required to run the local→unify→emit→select→diagnose path:
**no keys, no network.** Copy `.env.example` → `.env` to configure sources/keys.

## CLI

```bash
agentdata sources                                    # list sources + HF registry + local files
agentdata diagnose --report eval.json --out recipe.yaml
agentdata diagnose --scan ../some-skill/SKILL.md     # static introspection, no eval
agentdata build   --recipe examples/local_to_sft.yaml
agentdata build   --source local:sft_medical.jsonl --emit sft --size 200
agentdata stage   load --recipe examples/local_to_sft.yaml   # inspect one stage
```

## Architecture

```
sources/   pluggable INPUT connectors (local, hf, kaggle, physionet, github) + factory/router
unify/     detect_format → normalize → canonical DataItem
diagnose/  evalreport (parse scores) + introspect (scan target) + selector (rule table → Recipe)
generate/  llm providers (mock/anthropic) + synth + recombine + gepa
emit/      training-ready OUTPUT (sft, dpo, pretrain, chat, sharegpt, easydataset) + factory
select/    dedup + difficulty score + hard-biased curriculum
pipeline   run(recipe) / run_stage(stage, recipe) — the single execution path
report     provenance/audit manifest
```

**One contract, two frontends:** the CLI and any future `/agentdata` skill both
compile to a single `Recipe`, then call `pipeline.run(recipe)` — never two code
paths.

### Sources (Protocol + factory + router)

Every connector satisfies the `DataSource` Protocol (`list` / `fetch` /
`to_items` / `load`) and lazy-imports its dep with install guidance. A spec is
`"<source>[:<arg>]"` (`local:sft.jsonl`, `hf:locomo`, `gh:owner/repo`). A list of
specs fans out through a `SourceRouter` that concatenates, dedupes by item hash,
and keeps provenance. The HF registry ships named recipes (`hf:locomo`,
`hf:jackrong-claude-opus-distill`) plus generic `hf:<dataset_id>`. **PhysioNet
items carry `meta.redistributable=False` so emitters refuse to write gated raw
data to shareable outputs** (honors the MIMIC/PhysioNet DUA).

### Diagnosis → Recipe (the "auto" brain)

- `evalreport.parse` ingests a scores JSON (LoCoMo categories + math / reasoning /
  domain / intent / multimodal); below-threshold categories are gaps.
- `introspect.scan` statically walks a SKILL.md / MCP server / repo for missing
  capability signals when there's no eval report.
- `selector.select` maps gaps → `Recipe` via a rule table: weak `temporal` /
  `multi_hop` → SFT on LoCoMo-style recombined sessions; weak `math` / `reasoning`
  → GRPO/SFT on distilled reasoning sets; thin base knowledge → continue-pretrain.
  The GEPA principle is baked in: prefer small high-signal SFT + feedback traces.

### Emitters

Each validates required fields are non-empty strings (llm-trainer drops malformed
rows), writes UTF-8 JSONL one-object-per-line, and returns a manifest. Schemas
match the trainer ground-truth: SFT `{instruction,input,output}`, DPO
`{prompt,chosen,rejected}`, pretrain `{text}`, chat `{messages:[{role,content}]}`,
ShareGPT `{conversations:[{from,value}]}`; `easydataset` also emits the
LLaMA-Factory `dataset_info` block.

## Research basis

- **LoCoMo** ([arXiv 2402.17753](https://arxiv.org/abs/2402.17753)) — QA category
  taxonomy as diagnosis axes; persona + event-graph as the recombination template.
- **GEPA** ([arXiv 2507.19457](https://arxiv.org/abs/2507.19457)) — rich NL
  feedback/trajectory traces beat volume; keep sets small + high-signal.
- **Persona Hub** / **APIGen-MT** — persona-driven scaling + multi-turn synthesis
  → the recombination engine.
- **Claude-Opus distilled SFT** sets + **easy-dataset** export compatibility.

## Tests

```bash
python tests/test_smoke.py     # offline: no network, no keys
pytest tests/                  # same, pytest-discovered
RUN_LIVE=1 pytest tests/       # opt-in live HF/Kaggle/PhysioNet (skipped without creds)
```

The offline suite covers source Protocol conformance, the local source loading
`../dataset`, every emitter's JSONL validity, alpaca↔sharegpt↔chatml round-trips,
dedup/curriculum determinism, the DUA gate, the diagnosis→recipe selector,
generation determinism/immutability, and an end-to-end build with provenance.

## License

MIT
