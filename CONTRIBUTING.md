# Contributing to agentdata

Thanks for helping. agentdata stays small, offline-first, and dependency-light.

## Principles

- **One contract, two frontends.** The CLI and any skill compile to a `Recipe`,
  then call `pipeline.run(recipe)`. Never add a second execution path.
- **Offline-first.** Core (`local` source, unify, diagnose, emit, select) must
  run with no network and no keys. New deps go behind an extra + a lazy import
  with install guidance — never a top-level import in the core path.
- **Canonical `DataItem`.** Sources normalize *into* it; emitters consume *out of*
  it. Source/format specifics live in `meta`, not in new top-level fields.
- **Immutability.** Return new objects; don't mutate inputs (see `generate/gepa`).
- **Many small files.** High cohesion, low coupling; keep modules focused.

## Adding a source

1. Implement the `DataSource` Protocol (`list` / `fetch` / `to_items` / `load`)
   in `sources/<name>.py`; lazy-import the dep and raise `ImportError` with the
   install extra.
2. Register it in `sources/__init__.py:build_source` (+ an alias if useful).
3. Honor licensing: gate non-redistributable items with
   `meta["redistributable"] = False` (emitters refuse to write them).

## Adding an emitter

Subclass `BaseEmitter`, set `name` + `required`, implement `row()`. Register it in
`emit/__init__.py`. Match the trainer's exact JSONL schema.

## Tests

```bash
python tests/test_smoke.py    # offline; must stay green with no network/keys
pytest -q tests/test_smoke.py # same, pytest-discovered
RUN_LIVE=1 pytest tests/      # opt-in live source pulls
```

Add an offline assertion for any new source/emitter/stage. Live-only behavior
goes in `tests/test_live.py` (self-skips without `RUN_LIVE=1`).

## Commits

Conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`).
