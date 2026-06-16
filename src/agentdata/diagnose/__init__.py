"""Weakness diagnosis → Recipe selection.

`Diagnoser` is the facade: feed it an eval-scores report (or a target path to scan
statically), get back a Diagnosis, and turn that into a Recipe via the selector
rule table. Both inputs land in the same `select(...)`, so diagnosis has one
downstream path.
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..types import Diagnosis, Recipe
from . import evalreport, introspect, selector


class Diagnoser:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def from_report(self, report: dict[str, Any]) -> Diagnosis:
        return evalreport.parse(report, threshold=self.config.diagnose_threshold)

    def from_report_file(self, path: str) -> Diagnosis:
        return evalreport.parse_file(path, threshold=self.config.diagnose_threshold)

    def from_scan(self, path: str) -> Diagnosis:
        return introspect.scan(path, threshold=self.config.diagnose_threshold)

    def to_recipe(self, diagnosis: Diagnosis, out_dir: str | None = None,
                  name: str = "dataset") -> Recipe:
        return selector.select(diagnosis, out_dir=out_dir or self.config.out_dir, name=name,
                               rules=selector.load_rules(self.config.rules_path))

    def diagnose(self, *, report: dict[str, Any] | None = None,
                 report_file: str | None = None, scan_path: str | None = None,
                 name: str = "dataset") -> tuple[Diagnosis, Recipe]:
        """One-shot: pick whichever input is provided, return (Diagnosis, Recipe)."""
        if report is not None:
            dx = self.from_report(report)
        elif report_file is not None:
            dx = self.from_report_file(report_file)
        elif scan_path is not None:
            dx = self.from_scan(scan_path)
        else:
            raise ValueError("diagnose needs one of: report, report_file, scan_path")
        return dx, self.to_recipe(dx, name=name)


__all__ = ["Diagnoser", "evalreport", "introspect", "selector"]
