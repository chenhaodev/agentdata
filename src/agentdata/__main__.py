"""`agentdata` CLI — a thin front-end that compiles flags into a Recipe and calls
the pipeline core. Any future `/agentdata` skill compiles the same Recipe, so
there is exactly one execution path (CLAUDE.md one-contract rule).

Examples:
  agentdata sources                                   # list sources + HF registry
  agentdata diagnose --report eval.json --out recipe.yaml
  agentdata diagnose --scan ../some-skill/SKILL.md
  agentdata build --recipe examples/local_to_sft.yaml
  agentdata build --source local:sft_medical.jsonl --emit sft --size 200
  agentdata stage load --recipe examples/local_to_sft.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .builder import DatasetBuilder, load_recipe
from .config import Config
from .diagnose import Diagnoser
from .sources import build_source
from .sources.huggingface import REGISTRY
from .types import Recipe


def _cmd_sources(args, config: Config) -> int:
    if args.search:  # popularity-ranked Hub discovery (likes matter)
        from .sources.huggingface import HuggingFaceSource

        print(f"huggingface datasets for {args.search!r} (by likes):")
        for d in HuggingFaceSource(config).search(args.search, limit=args.limit):
            print(f"  likes={d['likes']:<5} dl={d['downloads']:<8} hf:{d['id']}")
        return 0
    print("sources: local, huggingface (hf), kaggle, physionet, github (gh)\n")
    print("huggingface named recipes (likes@registration):")
    for alias, info in REGISTRY.items():
        pop = f"likes={info.get('likes', '?')} dl={info.get('downloads', '?')}"
        print(f"  hf:{alias}  -> {info['dataset_id']}  [{pop}]  {info.get('tags', [])}")
    print("\nlocal files in DATA_DIR:")
    for f in build_source("local", config).list():
        print(f"  local:{f}")
    return 0


def _print_diagnosis(dx, recipe: Recipe) -> None:
    print("diagnosis:")
    for cap, s in sorted(dx.scores.items(), key=lambda kv: kv[1]):
        flag = "  GAP" if cap in dx.gaps else ""
        print(f"  {cap:14s} {s:.2f}{flag}")
    print(f"\ngaps: {dx.gaps or '(none)'}")
    print(f"\nrecipe -> regime={recipe.regime} emit={recipe.emit} "
          f"size={recipe.size} sources={recipe.sources}")
    if recipe.generate:
        print(f"        generate={recipe.generate}")


def _cmd_diagnose(args, config: Config) -> int:
    dgn = Diagnoser(config)
    if args.report:
        dx = dgn.from_report_file(args.report)
    elif args.scan:
        dx = dgn.from_scan(args.scan)
    else:
        print("error: diagnose needs --report <json> or --scan <path>", file=sys.stderr)
        return 2
    recipe = dgn.to_recipe(dx, out_dir=args.out_dir, name=args.name)
    _print_diagnosis(dx, recipe)
    if args.out:
        _write_recipe(recipe, args.out)
        print(f"\nwrote recipe -> {args.out}")
    return 0


def _write_recipe(recipe: Recipe, path: str) -> None:
    data = asdict(recipe)
    if path.endswith((".yaml", ".yml")):
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _recipe_from_args(args) -> Recipe:
    if args.recipe:
        return load_recipe(args.recipe)
    if not args.source:
        raise SystemExit("error: provide --recipe <file> or --source <spec>")
    return Recipe(
        sources=[s.strip() for s in args.source.split(",") if s.strip()],
        emit=args.emit, size=args.size, name=args.name, out_dir=args.out_dir,
        regime=args.regime, curriculum=not args.no_curriculum, dedup=not args.no_dedup,
    )


def _cmd_build(args, config: Config) -> int:
    recipe = _recipe_from_args(args)
    result = DatasetBuilder(config).build(recipe)
    m = result.manifest
    print(f"emit={m.emit}  wrote {m.count} samples -> {m.path}")
    print(f"  sources={m.sources}  licenses={m.licenses or '(none recorded)'}")
    print(f"  stats={m.stats}")
    print(f"manifest -> {result.report_path}")
    return 0


def _cmd_stage(args, config: Config) -> int:
    recipe = _recipe_from_args(args)
    out = DatasetBuilder(config).stage(args.stage, recipe)
    if isinstance(out, list):
        print(f"stage {args.stage!r}: {len(out)} items")
        for it in out[:5]:
            print(f"  {it!r}")
    else:  # emit returns a Manifest
        print(f"stage 'emit': wrote {out.count} -> {out.path}")
    return 0


def _add_recipe_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--recipe", help="recipe file (.yaml/.yml/.json)")
    p.add_argument("--source", help="comma list of source specs, e.g. local:sft.jsonl,hf:locomo")
    p.add_argument("--emit", default="sft", help="sft|dpo|pretrain|chat|sharegpt|easydataset")
    p.add_argument("--regime", default="sft", help="pretrain|continue_pretrain|sft|dpo|grpo")
    p.add_argument("--size", type=int, default=0, help="target sample count (0 = all)")
    p.add_argument("--name", default="dataset", help="output name")
    p.add_argument("--out-dir", default="out", dest="out_dir", help="output directory")
    p.add_argument("--no-curriculum", action="store_true", help="skip curriculum select")
    p.add_argument("--no-dedup", action="store_true", help="skip dedup")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentdata", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    srcp = sub.add_parser("sources", help="list sources + HF registry + local files")
    srcp.add_argument("--search", help="rank Hub datasets by likes for this query")
    srcp.add_argument("--limit", type=int, default=10, help="search result count")

    d = sub.add_parser("diagnose", help="diagnose weaknesses -> Recipe")
    d.add_argument("--report", help="eval scores JSON")
    d.add_argument("--scan", help="path to a SKILL.md / repo to introspect")
    d.add_argument("--out", help="write the chosen recipe to this file")
    d.add_argument("--name", default="dataset")
    d.add_argument("--out-dir", default="out", dest="out_dir")

    b = sub.add_parser("build", help="run the full pipeline for a recipe")
    _add_recipe_flags(b)

    s = sub.add_parser("stage", help="run a single pipeline stage for inspection")
    s.add_argument("stage", choices=["load", "generate", "dedup", "select", "emit"])
    _add_recipe_flags(s)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = Config.from_env()
    dispatch = {
        "sources": _cmd_sources,
        "diagnose": _cmd_diagnose,
        "build": _cmd_build,
        "stage": _cmd_stage,
    }
    return dispatch[args.cmd](args, config)


if __name__ == "__main__":
    raise SystemExit(main())
