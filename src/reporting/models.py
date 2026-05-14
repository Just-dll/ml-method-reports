from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ml_method_reports.reporting.types import ReportValue, TableRows


@dataclass(slots=True)
class ReportSection:
    title: str
    content: str | None = None
    table: TableRows | None = None
    code: str | None = None
    image_path: str | Path | None = None
    image_caption: str | None = None


@dataclass(slots=True)
class ExperimentReport:
    title: str
    subtitle: str | None = None
    metadata: dict[str, ReportValue] = field(default_factory=dict)
    sections: list[ReportSection] = field(default_factory=list)

