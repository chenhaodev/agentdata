"""PhysioNet source — DUA-gated clinical data (MIMIC, etc.).

Every item carries `meta.redistributable = False` so emitters refuse to write the
raw gated data into shareable outputs (honors the PhysioNet/MIMIC Data Use
Agreement — the meta-skill audit rule). Use it to *diagnose* and to drive
synthetic generation, not to redistribute corpora.

Auth via PHYSIONET_USER/PHYSIONET_PASSWORD; `requests` is lazy-imported.
"""

from __future__ import annotations

import os
from typing import Any

from ..config import Config
from ..types import DataItem
from .local import LocalSource

_BASE = "https://physionet.org/files/"


class PhysioNetSource:
    name = "physionet"

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def list(self, filters: dict[str, Any] | None = None) -> list[str]:
        return []  # project file trees are large; pass explicit relative file specs

    def _auth(self) -> tuple[str, str]:
        if not (self.config.physionet_user and self.config.physionet_password):
            raise PermissionError(
                "PhysioNet source needs PHYSIONET_USER/PHYSIONET_PASSWORD and a signed "
                "Data Use Agreement for the requested project."
            )
        return self.config.physionet_user, self.config.physionet_password

    def fetch(self, spec: str) -> dict[str, Any]:
        try:
            import requests  # lazy: optional dep
        except ImportError as e:
            raise ImportError(
                "PhysioNet source needs `requests`. Install: pip install 'agentdata[physionet]'"
            ) from e
        auth = self._auth()
        url = spec if spec.startswith("http") else _BASE + spec.lstrip("/")
        dest = os.path.join(self.config.cache_dir, "physionet", os.path.basename(spec))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        resp = requests.get(url, auth=auth, timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        return {"spec": spec, "path": dest}

    def to_items(self, raw: dict[str, Any]) -> list[DataItem]:
        items = LocalSource(self.config).load(raw["path"])
        for it in items:
            # gate every item: non-redistributable per DUA
            it.meta = {**it.meta, "source": self.name, "spec": raw.get("spec"),
                       "redistributable": False, "license": "physionet-dua"}
        return items

    def load(self, spec: str) -> list[DataItem]:
        return self.to_items(self.fetch(spec))
