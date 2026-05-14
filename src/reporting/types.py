from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ml_method_reports.reporting.models import ExperimentReport

FeatureMatrix: TypeAlias = pd.DataFrame | np.ndarray | Sequence[Sequence[Any]]
TargetVector: TypeAlias = pd.Series | np.ndarray | Sequence[Any]
PredictionVector: TypeAlias = np.ndarray | Sequence[Any]
ReportMetadata: TypeAlias = Mapping[str, Any]
ReportOptions: TypeAlias = Mapping[str, Any]
PathLike: TypeAlias = str | Path
ClassLabel: TypeAlias = Hashable
ReportValue: TypeAlias = object
TableRow: TypeAlias = dict[str, ReportValue]
TableRows: TypeAlias = list[TableRow]
ScalingParams: TypeAlias = Mapping[str, Any]


@runtime_checkable
class PredictiveModel(Protocol):
    def predict(self, X: FeatureMatrix) -> PredictionVector:
        ...


@runtime_checkable
class ParametrizedModel(Protocol):
    def get_params(self) -> Mapping[str, Any]:
        ...


class ReportBuilderObject(Protocol):
    def build(self) -> ExperimentReport:
        ...
