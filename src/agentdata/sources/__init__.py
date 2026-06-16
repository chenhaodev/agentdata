"""Source adapters + the factory/router that turns specs into DataItems.

A *spec* is `"<source>[:<arg>]"` — e.g. `local:sft_medical.jsonl`, `hf:locomo`,
`gh:owner/repo`, `kaggle:owner/dataset`. `build_source` maps the source name to a
connector (lazy deps); `load_sources` fans a list of specs out through a
SourceRouter that concatenates, dedupes by item hash, and keeps provenance.

Mirrors agentmem's backends factory + RouterBackend.
"""

from __future__ import annotations

from ..config import Config
from ..types import DataItem
from .base import DataSource

# source-name aliases
_ALIASES = {"hf": "huggingface", "gh": "github"}


def build_source(name: str, config: Config | None = None) -> DataSource:
    """Build one source connector by name (adapters import their deps lazily)."""
    config = config or Config()
    name = _ALIASES.get(name, name)
    if name == "local":
        from .local import LocalSource

        return LocalSource(config)
    if name == "huggingface":
        from .huggingface import HuggingFaceSource

        return HuggingFaceSource(config)
    if name == "kaggle":
        from .kaggle import KaggleSource

        return KaggleSource(config)
    if name == "physionet":
        from .physionet import PhysioNetSource

        return PhysioNetSource(config)
    if name == "github":
        from .github import GitHubSource

        return GitHubSource(config)
    raise ValueError(
        f"Unknown source {name!r}. Expected one of: "
        "local, huggingface (hf), kaggle, physionet, github (gh)."
    )


def _parse_spec(spec: str) -> tuple[str, str]:
    """`local:sft.jsonl` -> ("local", "sft.jsonl"); `local` -> ("local", "")."""
    name, _, arg = spec.partition(":")
    return name.strip(), arg.strip()


class SourceRouter:
    """Fan a list of specs out across connectors; concat + dedupe + keep provenance.

    Realizes the multi-source "pull from anywhere, unify the format" goal. Read
    resilience: one spec failing is recorded in `errors` and skipped, the rest
    still load — unless *every* spec fails, which re-raises.
    """

    def __init__(self, specs: list[str], config: Config | None = None):
        if not specs:
            raise ValueError("SourceRouter needs at least one source spec")
        self.specs = specs
        self.config = config or Config()
        self.errors: list[tuple[str, Exception]] = []

    def _load_one(self, spec: str) -> list[DataItem]:
        name, arg = _parse_spec(spec)
        source = build_source(name, self.config)
        if arg:
            return source.load(arg)
        # no arg: local loads every discoverable file; others need an explicit arg
        items: list[DataItem] = []
        for found in source.list():
            items.extend(source.load(found))
        return items

    def _safe_load(self, spec: str) -> list[DataItem] | None:
        try:
            return self._load_one(spec)
        except Exception as e:  # resilience: record + skip; caller keeps the rest
            self.errors.append((spec, e))
            return None

    def _load_all(self) -> list[list[DataItem] | None]:
        """Load every spec, results aligned with self.specs. Specs are independent
        network/disk I/O, so fan them out across threads; `map` preserves order so
        the deduped output stays deterministic. One spec ⇒ no thread overhead."""
        if len(self.specs) == 1 or self.config.load_workers <= 1:
            return [self._safe_load(s) for s in self.specs]
        from concurrent.futures import ThreadPoolExecutor

        workers = min(self.config.load_workers, len(self.specs))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(self._safe_load, self.specs))

    def load(self) -> list[DataItem]:
        results = self._load_all()
        out: list[DataItem] = []
        seen: set[str] = set()
        failures = 0
        for items in results:  # spec order → dedup keeps first occurrence, deterministic
            if items is None:
                failures += 1
                continue
            for it in items:
                h = it.hash()
                if h in seen:
                    continue
                seen.add(h)
                out.append(it)
        if failures == len(self.specs):  # per-call count (idempotent across load() calls)
            raise self.errors[-1][1]  # every spec failed — surface it
        return out


def load_sources(specs: list[str], config: Config | None = None) -> list[DataItem]:
    """Top-level convenience: load + dedupe all specs into one item list."""
    return SourceRouter(specs, config).load()


__all__ = [
    "DataSource",
    "build_source",
    "load_sources",
    "SourceRouter",
]
