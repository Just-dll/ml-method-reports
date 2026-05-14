from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ml_method_reports.reporting.models import ExperimentReport


class ReportBuilder(ABC):
    @abstractmethod
    def build(self) -> ExperimentReport:
        ...


class ReportAssetBuilder(ABC):
    @abstractmethod
    def build_assets(self, assets_dir: Path) -> dict[str, str]:
        ...


# Future report builders can live next to this module:
# TODO: reporting/builders/clustering.py
# TODO: reporting/builders/feature_selection.py
# TODO: reporting/builders/supervised_comparison.py

