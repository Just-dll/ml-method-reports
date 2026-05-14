from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ml_method_reports.reporting.types import (
    FeatureMatrix,
    PredictionVector,
    ReportMetadata,
    ReportOptions,
    TargetVector,
)


@dataclass(slots=True)
class ReportContext:
    model: object
    model_name: str
    X_train: FeatureMatrix | None
    X_test: FeatureMatrix
    y_train: TargetVector | None
    y_test: TargetVector | None
    predictions: PredictionVector | None
    feature_names: list[str] | None
    target_name: str
    dataset_source: str
    output_dir: Path
    assets_dir: Path | None
    metadata: ReportMetadata = field(default_factory=dict)
    options: ReportOptions = field(default_factory=dict)
