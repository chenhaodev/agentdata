"""Offline smoke tests for agentdata. Run: python tests/test_smoke.py

No network, no API key, no external services — exercises source Protocol
conformance, unify round-trips, every emitter, dedup/curriculum determinism, the
DUA gate, the diagnosis→recipe selector, generation, and an end-to-end build.
Plain asserts so it runs without pytest; `pytest tests/` also discovers test_*.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentdata import Config, DatasetBuilder, Diagnoser, Recipe  # noqa: E402
from agentdata.emit import build_emitter  # noqa: E402
from agentdata.emit.base import Emitter  # noqa: E402
from agentdata.generate import (  # noqa: E402
    attach_feedback, get_provider, keep_high_signal, recombine, synth,
)
from agentdata.select import curriculum_select, dedup, score_difficulty  # noqa: E402
from agentdata.sources import build_source, load_sources  # noqa: E402
from agentdata.sources.base import DataSource  # noqa: E402
from agentdata.types import KIND_MESSAGES, KIND_QA, KIND_TEXT, DataItem  # noqa: E402
from agentdata.unify.normalize import normalize_row  # noqa: E402


def _msg_item(user="what is X?", asst="X is a thing.", source="t"):
    return DataItem(KIND_MESSAGES,
                    messages=[{"role": "user", "content": user},
                              {"role": "assistant", "content": asst}],
                    meta={"source": source})


# -- sources -----------------------------------------------------------------

def test_source_protocol_conformance():
    for name in ("local", "huggingface", "kaggle", "physionet", "github"):
        src = build_source(name, Config())
        assert isinstance(src, DataSource), f"{name} not a DataSource"
    print("ok  every source satisfies the DataSource Protocol")


def test_unknown_source_errors_clearly():
    try:
        build_source("nope", Config())
        assert False, "expected ValueError"
    except ValueError as e:
        assert "physionet" in str(e)  # error lists known sources
    print("ok  unknown source raises a helpful ValueError")


def test_source_lazy_dep_messages():
    """Optional-dep sources fail with install guidance, not a raw ImportError."""
    for name, hint in (("huggingface", "agentdata[hf]"),):
        try:
            import datasets  # noqa: F401

            print(f"skip {name}: dep installed; message path not exercised")
            continue
        except ImportError:
            pass
        try:
            build_source(name, Config()).fetch("x")
            assert False, "expected ImportError"
        except ImportError as e:
            assert hint in str(e)
    print("ok  optional-dep sources give install guidance")


def test_local_source_loads_dataset_dir():
    """If ../dataset exists, the local source loads & normalizes real corpora."""
    cfg = Config(data_dir=os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
    src = build_source("local", cfg)
    files = src.list()
    if "sft_medical.jsonl" not in files:
        print("skip local dataset: ../dataset/sft_medical.jsonl not present")
        return
    items = src.load("sft_medical.jsonl")
    assert items and all(it.meta["source"] == "local" for it in items[:5])
    assert items[0].kind == KIND_MESSAGES
    print(f"ok  local source loaded {len(items)} items from ../dataset")


def test_router_dedup_across_specs():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.write('{"instruction": "dup q", "input": "", "output": "ans"}\n')
            f.write('{"instruction": "dup q", "input": "", "output": "ans"}\n')
            f.write('{"instruction": "uniq", "input": "", "output": "ans2"}\n')
        items = load_sources([f"local:{p}", f"local:{p}"], Config(data_dir=d))
    keys = {it.content_key() for it in items}
    assert len(items) == len(keys) == 2, items  # deduped across duplicate specs
    print("ok  source router dedupes across specs by item hash")


def test_parallel_load_is_resilient_and_ordered():
    """Parallel multi-source load: a bad spec is skipped (others still load), output
    stays in spec order; all-fail re-raises."""
    from agentdata.sources import SourceRouter

    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.jsonl")
        b = os.path.join(d, "b.jsonl")
        with open(a, "w", encoding="utf-8") as f:
            f.write('{"text": "alpha"}\n')
        with open(b, "w", encoding="utf-8") as f:
            f.write('{"text": "bravo"}\n')
        cfg = Config(data_dir=d, load_workers=8)
        # good, bad (missing file), good → two items, in spec order, error recorded
        r = SourceRouter([f"local:{a}", "local:/no/such.jsonl", f"local:{b}"], cfg)
        items = r.load()
        assert [it.text for it in items] == ["alpha", "bravo"], items
        assert len(r.errors) == 1
        # every spec fails → raise
        allbad = SourceRouter(["local:/x.jsonl", "local:/y.jsonl"], cfg)
        try:
            allbad.load()
            assert False, "expected raise when all specs fail"
        except Exception:
            pass
    print("ok  parallel source load is resilient, ordered, and fails loudly when total")


# -- unify -------------------------------------------------------------------

def test_unify_roundtrip_formats():
    alp = normalize_row({"instruction": "Q", "input": "ctx", "output": "A"})
    sg = normalize_row({"conversations": [{"from": "human", "value": "Q"},
                                          {"from": "gpt", "value": "A"}]})
    cm = normalize_row({"messages": [{"role": "user", "content": "Q"},
                                     {"role": "assistant", "content": "A"}]})
    qa = normalize_row({"question": "Q", "answer": "A"})
    pl = normalize_row({"text": "some corpus text"})
    assert alp.kind == sg.kind == cm.kind == KIND_MESSAGES
    assert qa.kind == KIND_QA and pl.kind == KIND_TEXT
    # all message-shaped formats normalize to the same last-turn content
    assert sg.messages[-1]["content"] == cm.messages[-1]["content"] == "A"
    assert sg.messages[0]["role"] == "user"  # human -> user
    assert normalize_row({"instruction": "", "output": ""}) is None  # empty rejected
    print("ok  unify round-trips alpaca/sharegpt/chatml/qa/plain")


# -- emit --------------------------------------------------------------------

def test_each_emitter_valid_jsonl():
    items = [_msg_item(), DataItem(KIND_QA, question="q?", answer="a", meta={"source": "t"}),
             DataItem(KIND_TEXT, text="corpus body text here", meta={"source": "t"})]
    items_dpo = [_msg_item(), DataItem(KIND_MESSAGES,
                 messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "good"}],
                 meta={"source": "t", "rejected": "bad"})]
    cases = {"sft": items, "pretrain": items, "chat": items, "sharegpt": items,
             "easydataset": items, "devset": items, "dpo": items_dpo}
    with tempfile.TemporaryDirectory() as d:
        for emit, data in cases.items():
            emitter = build_emitter(emit)
            assert isinstance(emitter, Emitter)
            path = os.path.join(d, f"{emit}.jsonl")
            m = emitter.emit(data, path)
            assert m.count >= 1, f"{emit} wrote nothing"
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            assert len(rows) == m.count
            for r in rows:  # required fields present and non-empty
                for k in emitter.required:
                    v = r[k]
                    if isinstance(v, list):
                        assert v and v[-1].get("content") or v[-1].get("value")
                    else:
                        assert isinstance(v, str) and v.strip()
    print("ok  every emitter writes valid JSONL with non-empty required fields")


def test_easydataset_emits_dataset_info():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ed.jsonl")
        m = build_emitter("easydataset").emit([_msg_item()], path)
        info_path = m.stats["dataset_info_path"]
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        entry = info["ed"]
        assert entry["columns"] == {"prompt": "instruction", "query": "input", "response": "output"}
    print("ok  easydataset emits a LLaMA-Factory dataset_info block")


def test_devset_emitter_for_prompt_optimizers():
    """The devset emitter backs the prompt-opt path: {input,target} rows + a metric
    in the manifest (the labeled half GEPA/DSPy consume)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dev.jsonl")
        m = build_emitter("devset").emit([_msg_item(user="2+2?", asst="4")], path)
        assert m.count == 1 and m.stats["metric"] == "exact_match"
        with open(path, encoding="utf-8") as f:
            row = json.loads(f.readline())
        assert set(row) == {"input", "target"} and row["target"] == "4"
    print("ok  devset emitter yields {input,target} + metric for prompt optimizers")


def test_dua_gate_blocks_shareable_emit():
    gated = _msg_item(source="physionet")
    gated.meta["redistributable"] = False
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "g.jsonl")
        m = build_emitter("sft").emit([gated, _msg_item()], path)
    assert m.count == 1, "gated item leaked into output"
    assert m.stats["gated_skipped"] == 1
    print("ok  redistributable=False blocks shareable emit (DUA honored)")


# -- select ------------------------------------------------------------------

def test_dedup_and_curriculum_determinism():
    items = [_msg_item(user=f"q{i}", asst="<think>" + "w " * (i * 30) + "</think> a")
             for i in range(20)] + [_msg_item(user="q0")]  # one dup of q0
    dd = dedup(items)
    assert len(dd) == 20, "dedup miscount"
    a = curriculum_select(dd, n_target=8, seed=42)
    b = curriculum_select(dd, n_target=8, seed=42)
    assert [x.content_key() for x in a] == [x.content_key() for x in b], "not deterministic"
    # easy→hard ordering
    scores = [score_difficulty(x) for x in a]
    assert scores == sorted(scores), "curriculum not easy→hard"
    print("ok  dedup + curriculum select are deterministic and curriculum-sorted")


# -- diagnose ----------------------------------------------------------------

def test_diagnosis_selects_expected_recipe():
    report = {"scores": {"single_hop": 0.85, "multi_hop": 0.4, "temporal": 0.35,
                         "math": 0.3, "reasoning": 0.5}}
    dx, recipe = Diagnoser(Config(diagnose_threshold=0.6)).diagnose(report=report)
    assert set(dx.gaps) == {"multi_hop", "temporal", "math", "reasoning"}, dx.gaps
    # math gap -> GRPO regime wins by priority; LoCoMo + distilled sources present
    assert recipe.regime == "grpo", recipe.regime
    assert "hf:locomo" in recipe.sources and "hf:jackrong-claude-opus-distill" in recipe.sources
    assert recipe.generate.get("recombine") and recipe.generate.get("gepa")
    assert recipe.meta["gaps_addressed"] == dx.gaps
    print("ok  diagnosis maps temporal/multi_hop/math/reasoning -> correct Recipe")


def test_diagnosis_introspect_scan():
    with tempfile.TemporaryDirectory() as d:
        skill = os.path.join(d, "SKILL.md")
        with open(skill, "w", encoding="utf-8") as f:
            f.write("# A skill that calls tools via MCP and reads a domain corpus.\n")
        dx = Diagnoser().from_scan(skill)
    assert "tool_use" not in dx.gaps and "domain" not in dx.gaps  # signals present
    assert "reasoning" in dx.gaps  # no <think>/reasoning signal
    print("ok  introspect scan flags missing capabilities from a SKILL.md")


def test_diagnose_memory_benchmark():
    """A memory-retrieval benchmark (agentmem's {hit1,hit3,clean,mrr} per backend) is
    diagnosed: weakest backend chosen, timings ignored, → LoCoMo-recombined recipe."""
    report = [
        {"backend": "vector", "ok": True, "stored": 12, "write_s": 0.3, "query_ms": 4.1,
         "hit1": 0.42, "hit3": 0.55, "clean": 0.38, "mrr": 0.49},
        {"backend": "mem0", "ok": True, "stored": 12, "write_s": 6.2, "query_ms": 88.0,
         "hit1": 0.58, "hit3": 0.71, "clean": 0.50, "mrr": 0.63},
    ]
    dx, recipe = Diagnoser(Config(diagnose_threshold=0.6)).diagnose(report=report)
    assert "memory" in dx.gaps, dx.gaps
    assert dx.scores["hit1"] == 0.42, "should pick the weakest backend (vector)"
    assert "stored" not in dx.scores and "query_ms" not in dx.scores  # timings/counts ignored
    assert recipe.regime == "sft" and recipe.emit == "chat"
    assert recipe.sources == ["hf:locomo"] and recipe.generate.get("recombine")
    print("ok  diagnose ingests an agentmem memory benchmark -> LoCoMo recipe")


# -- extensibility (fill-in YAML, no code) -----------------------------------

def test_registry_yaml_extends_without_code():
    from agentdata.sources.huggingface import HuggingFaceSource

    with tempfile.TemporaryDirectory() as d:
        reg = os.path.join(d, "registry.yaml")
        with open(reg, "w", encoding="utf-8") as f:
            f.write("acme-set:\n  dataset_id: acme/data\n  file: data/t.jsonl\n"
                    "  format: jsonl\n  tags: [demo]\n")
        src = HuggingFaceSource(Config(registry_path=reg))
        assert "acme-set" in src.list()                       # new recipe, no code edit
        assert src._resolve("acme-set")["dataset_id"] == "acme/data"
        # built-ins still present
        assert "locomo" in src.list()
    print("ok  registry YAML adds HF recipes without touching Python")


def test_rules_yaml_extends_and_overrides():
    from agentdata.diagnose import selector

    with tempfile.TemporaryDirectory() as d:
        rules = os.path.join(d, "rules.yaml")
        with open(rules, "w", encoding="utf-8") as f:
            f.write("safety:\n  regime: sft\n  dataset_types: [safety]\n"
                    "  sources: [local]\n  emit: chat\n  generate: {synth: true}\n")
        dx, recipe = Diagnoser(Config(rules_path=rules)).diagnose(
            report={"scores": {"safety": 0.2}})
        assert "safety" in dx.gaps
        assert recipe.emit == "chat" and recipe.generate.get("synth")
        # a path-less load is just the built-ins (no surprise side effects)
        assert "safety" not in selector.load_rules("")
    print("ok  rules YAML adds/overrides diagnosis rules without touching Python")


# -- generate ----------------------------------------------------------------

def test_generate_offline_deterministic():
    provider = get_provider("mock")
    seeds = [DataItem(KIND_TEXT, text="Hypertension is high blood pressure that is chronic and common.",
                      meta={"source": "t", "tags": ["cardio"]}),
             DataItem(KIND_TEXT, text="Hypertension raises chronic cardiovascular risk over time.",
                      meta={"source": "t", "tags": ["cardio"]})]
    s1 = synth(seeds, provider)
    s2 = synth(seeds, provider)
    assert s1 and [x.messages for x in s1] == [x.messages for x in s2], "synth not deterministic"
    assert "<think>" in s1[0].messages[-1]["content"]  # reasoning trace present
    rc = recombine(seeds, provider)
    assert rc and rc[0].meta["gen"] == "recombine" and rc[0].meta["n_subjects"] == 2
    fb = attach_feedback(s1)
    assert all("feedback" in x.meta and "trajectory" in x.meta for x in fb)
    assert seeds[0].meta.get("feedback") is None, "attach_feedback mutated input"
    assert len(keep_high_signal(fb, cap=1)) == 1
    print("ok  synth/recombine/gepa are offline, deterministic, immutable")


def test_multi_agent_recombine_and_roundtrip():
    """multi_agent recombination assigns a distinct agent role per subject, and the
    chat/sharegpt emitters preserve those roles end-to-end."""
    subjects = [
        DataItem(KIND_MESSAGES, messages=[{"role": "user", "content": "plan the trip together"},
                 {"role": "assistant", "content": "I propose a 3-day route"}],
                 meta={"source": "t", "tags": ["travel"]}),
        DataItem(KIND_MESSAGES, messages=[{"role": "user", "content": "plan the trip together"},
                 {"role": "assistant", "content": "I'd add a budget check"}],
                 meta={"source": "t", "tags": ["travel"]}),
    ]
    rc = recombine(subjects, get_provider("mock"), multi_agent=True)
    assert rc, "no multi-agent transcript produced"
    roles = {m["role"] for m in rc[0].messages}
    assert {"agent1", "agent2"} <= roles, roles  # distinct agents per subject
    assert rc[0].meta["multi_agent"] is True
    # roles survive both chat and sharegpt exports
    chat_row = build_emitter("chat").row(rc[0])
    assert {"agent1", "agent2"} <= {m["role"] for m in chat_row["messages"]}
    sg_row = build_emitter("sharegpt").row(rc[0])
    assert {"agent1", "agent2"} <= {t["from"] for t in sg_row["conversations"]}
    print("ok  multi-agent recombination + role-preserving chat/sharegpt export")


# -- end to end --------------------------------------------------------------

def test_end_to_end_build():
    with tempfile.TemporaryDirectory() as d:
        corpus = os.path.join(d, "c.jsonl")
        with open(corpus, "w", encoding="utf-8") as f:
            for i in range(12):
                f.write(json.dumps({"instruction": f"Question number {i} about medicine "
                                    "with enough words to pass filters", "input": "",
                                    "output": f"Answer {i}"}) + "\n")
        cfg = Config(data_dir=d, out_dir=os.path.join(d, "out"))
        recipe = Recipe(sources=[f"local:{corpus}"], emit="sft", size=6, name="e2e")
        result = DatasetBuilder(cfg).build(recipe)
        assert result.manifest.count == 6, result.manifest.count
        assert os.path.exists(result.manifest.path)
        assert os.path.exists(result.report_path)
        with open(result.manifest.path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert all({"instruction", "input", "output"} <= set(r) for r in rows)
        with open(result.report_path, encoding="utf-8") as f:
            report = json.load(f)
        assert report["provenance"]["by_source"]["local"] >= 6
    print("ok  end-to-end build emits SFT JSONL + manifest with provenance")


def test_dpo_self_sufficient():
    """`--emit dpo` synthesizes preference pairs on its own (no pre-existing rejected):
    every row has prompt/chosen/rejected with chosen != rejected."""
    with tempfile.TemporaryDirectory() as d:
        corpus = os.path.join(d, "c.jsonl")
        with open(corpus, "w", encoding="utf-8") as f:
            for i in range(8):
                f.write(json.dumps({"instruction": f"Explain medical topic {i} in detail "
                                    "with enough words here", "input": "",
                                    "output": f"<think>reason {i}</think> A full answer number {i} "
                                    "with several words of substance"}) + "\n")
        cfg = Config(data_dir=d, out_dir=os.path.join(d, "out"))
        recipe = Recipe(sources=[f"local:{corpus}"], emit="dpo", name="dpo", dedup=False)
        result = DatasetBuilder(cfg).build(recipe)
        assert result.manifest.count > 0, "DPO produced an empty dataset"
        with open(result.manifest.path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert all({"prompt", "chosen", "rejected"} == set(r) for r in rows)
        assert all(r["chosen"].strip() != r["rejected"].strip() for r in rows)
        assert all("<think>" not in r["rejected"] for r in rows)  # degraded: reasoning stripped
    print("ok  --emit dpo synthesizes non-empty preference pairs (chosen != rejected)")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    main()
