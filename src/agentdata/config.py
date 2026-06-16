"""Configuration loaded from environment / .env.

Mirrors agentmem's Config: a frozen-ish dataclass with a from_env() classmethod.
Every field has an offline-safe default so the package imports and runs the
local→unify→emit→select path with no keys and no network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()  # populate os.environ from .env if present
except Exception:  # python-dotenv optional at import time
    pass


@dataclass
class Config:
    # --- teacher LLM (synthetic generation) ---
    llm_provider: str = "auto"  # auto | mock | anthropic
    llm_model: str = "claude-opus-4-8"
    anthropic_api_key: str = ""

    # --- source selection ---
    # single ("local") or a "+"/"," list ("local+huggingface") for cross-source fan-out
    source: str = "local"

    # --- paths ---
    data_dir: str = "../dataset"  # local corpora root
    out_dir: str = "out"  # emitted datasets + manifests
    cache_dir: str = ".data/cache"  # downloaded raw source cache

    # --- huggingface source ---
    hf_token: str = ""
    hf_cache_dir: str = ""

    # --- kaggle source ---
    kaggle_username: str = ""
    kaggle_key: str = ""

    # --- physionet source (DUA-gated) ---
    physionet_user: str = ""
    physionet_password: str = ""

    # --- github source ---
    github_token: str = ""

    # --- diagnosis ---
    diagnose_threshold: float = 0.6  # below this per-category score = a gap

    # --- extension points (fill-in YAML, no code) ---
    # drop a YAML to add HF dataset recipes / tune the gap→recipe rules without editing Python
    registry_path: str = ""  # extra HF registry entries (see examples/registry.yaml)
    rules_path: str = ""  # extra/overriding selector rules (see examples/rules.yaml)

    # --- performance ---
    load_workers: int = 8  # parallel threads for multi-source loading (I/O-bound)

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "auto"),
            llm_model=os.getenv("LLM_MODEL", "claude-opus-4-8"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            source=os.getenv("SOURCE", "local"),
            data_dir=os.getenv("DATA_DIR", "../dataset"),
            out_dir=os.getenv("OUT_DIR", "out"),
            cache_dir=os.getenv("CACHE_DIR", ".data/cache"),
            hf_token=os.getenv("HF_TOKEN", ""),
            hf_cache_dir=os.getenv("HF_CACHE_DIR", ""),
            kaggle_username=os.getenv("KAGGLE_USERNAME", ""),
            kaggle_key=os.getenv("KAGGLE_KEY", ""),
            physionet_user=os.getenv("PHYSIONET_USER", ""),
            physionet_password=os.getenv("PHYSIONET_PASSWORD", ""),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            diagnose_threshold=float(os.getenv("DIAGNOSE_THRESHOLD", "0.6")),
            registry_path=os.getenv("AGENTDATA_REGISTRY", ""),
            rules_path=os.getenv("AGENTDATA_RULES", ""),
            load_workers=int(os.getenv("LOAD_WORKERS", "8")),
        )
