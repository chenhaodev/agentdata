"""GitHub source — fetch code / SKILL.md / MCP-server corpora as plain-text items.

Spec form: `<owner>/<repo>` (default branch) or `<owner>/<repo>@<ref>`. Pulls the
repo tarball via the API and emits one KIND_TEXT item per matching text file, so
code/doc corpora flow straight into a continue-pretrain emit.
"""

from __future__ import annotations

import io
import os
import tarfile
from typing import Any

from ..config import Config
from ..types import KIND_TEXT, DataItem

_TEXT_EXTS = (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".js", ".ts", ".rs", ".go")


class GitHubSource:
    name = "github"

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def list(self, filters: dict[str, Any] | None = None) -> list[str]:
        return []  # pass explicit <owner>/<repo> specs

    def fetch(self, spec: str) -> dict[str, Any]:
        try:
            import requests  # lazy: optional dep
        except ImportError as e:
            raise ImportError(
                "GitHub source needs `requests`. Install: pip install 'agentdata[github]'"
            ) from e
        repo, _, ref = spec.partition("@")
        url = f"https://api.github.com/repos/{repo}/tarball/{ref}" if ref \
            else f"https://api.github.com/repos/{repo}/tarball"
        headers = {"Accept": "application/vnd.github+json"}
        if self.config.github_token:
            headers["Authorization"] = f"Bearer {self.config.github_token}"
        resp = requests.get(url, headers=headers, timeout=120)
        resp.raise_for_status()
        files: list[tuple[str, str]] = []
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(_TEXT_EXTS):
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                try:
                    text = fh.read().decode("utf-8", errors="ignore")
                except Exception:
                    continue
                files.append((member.name, text))
        return {"spec": spec, "files": files}

    def to_items(self, raw: dict[str, Any]) -> list[DataItem]:
        items: list[DataItem] = []
        for name, text in raw.get("files", []):
            if not text.strip():
                continue
            items.append(DataItem(
                kind=KIND_TEXT, text=text,
                meta={"source": self.name, "spec": raw.get("spec"),
                      "path": os.path.relpath(name, name.split("/")[0]),
                      "license": "see-repo"},
            ))
        return items

    def load(self, spec: str) -> list[DataItem]:
        return self.to_items(self.fetch(spec))
