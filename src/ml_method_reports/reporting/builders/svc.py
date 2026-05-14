from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

from ml_method_reports.reporting.builders.base import ReportAssetBuilder, ReportBuilder
from ml_method_reports.reporting.builders.sklearn_common import (
    as_2d_array,
    build_bar_asset,
    confusion_rows,
    error_analysis_text,
    evaluation_rows,
    feature_names,
    feature_score_rows,
    model_params_rows,
    predictions,
    preprocessing_section,
    probability_rows,
    score_rows,
    selected_index,
    selected_query,
    total_errors,
)
from ml_method_reports.reporting.models import ExperimentReport, ReportSection
from ml_method_reports.reporting.types import (
    FeatureMatrix,
    PredictionVector,
    ScalingParams,
    TableRows,
    TargetVector,
)


@dataclass(slots=True)
class SvcReportInput:
    model: SVC
    X_train: FeatureMatrix
    X_test: FeatureMatrix
    y_train: TargetVector
    y_test: TargetVector
    feature_names: list[str] | None = None
    predictions: PredictionVector | None = None
    dataset_source: str = "classification dataset"
    target_name: str = "target"
    selected_sample_index: int = 0
    scaling_method: str = "none"
    scaling_params: ScalingParams | None = None


class SvcReportBuilder(ReportBuilder, ReportAssetBuilder):
    def __init__(self, report_input: SvcReportInput, assets_dir: Path | None = None) -> None:
        self._input = report_input
        self._assets_dir = assets_dir

    def build(self) -> ExperimentReport:
        assets = self.build_assets(self._assets_dir) if self._assets_dir else {}
        return self._build_report(assets)

    def build_assets(self, assets_dir: Path | None = None) -> dict[str, str]:
        if assets_dir is None:
            if self._assets_dir is None:
                raise ValueError("assets_dir must be provided to build report assets.")
            assets_dir = self._assets_dir
        assets: dict[str, str] = {}
        assets.update(build_bar_asset(_support_rows(self._input.model), assets_dir, "svc_support_vectors.png", label_key="class", value_key="support_vectors", title="SVC Support Vectors"))
        coef_rows = _linear_coefficient_rows(self._input.model, feature_names(self._input.X_train, self._input.feature_names))
        assets.update(build_bar_asset(coef_rows, assets_dir, "svc_linear_coefficients.png", label_key="feature", value_key="coefficient", title="Linear SVC Coefficients"))
        return assets

    def _build_report(self, assets: dict[str, str]) -> ExperimentReport:
        data = self._input
        X_test = as_2d_array(data.X_test)
        names = feature_names(data.X_train, data.feature_names)
        pred = predictions(data.model, data.X_test, data.predictions)
        index = selected_index(data.selected_sample_index, X_test.shape[0])
        query = selected_query(data.X_test, X_test, index)
        confusion = confusion_rows(data.y_test, pred)
        errors = total_errors(data.y_test, pred)
        accuracy = float(np.mean(np.asarray(data.y_test) == pred))
        coef_rows = _linear_coefficient_rows(data.model, names)

        sections = [
            ReportSection(title="1. Experiment Overview", table=[
                {"item": "dataset source", "value": data.dataset_source},
                {"item": "train samples", "value": int(as_2d_array(data.X_train).shape[0])},
                {"item": "test samples", "value": int(X_test.shape[0])},
                {"item": "features", "value": len(names)},
                {"item": "selected sample index", "value": index},
            ]),
            preprocessing_section(data.scaling_method, data.scaling_params, names),
            ReportSection(title="3. SVC Parameters", table=model_params_rows(data.model, ["C", "kernel", "gamma", "degree", "probability"])),
            ReportSection(title="4. Kernel Interpretation Notes", content=_kernel_note(data.model)),
            ReportSection(title="5. Support Vector Summary", table=_support_rows(data.model), image_path=assets.get("svc_support_vectors")),
            ReportSection(title="6. Selected Prediction Decision Scores", table=score_rows(data.model, query) + probability_rows(data.model, query)),
            ReportSection(title="7. Linear Kernel Coefficients", content=_linear_note(data.model), table=coef_rows, image_path=assets.get("svc_linear_coefficients")),
            ReportSection(title="8. Evaluation", table=evaluation_rows(data.y_test, pred)),
            ReportSection(title="9. Confusion Matrix", table=confusion),
            ReportSection(title="10. Error Analysis", content=error_analysis_text(confusion)),
            ReportSection(title="11. Analysis Summary", content=f"SVC achieved {accuracy:.3f} accuracy with {errors} error(s). The report exposes support vectors and decision scores; nonlinear kernels are intentionally not presented as simple feature rules."),
        ]
        return ExperimentReport(
            title="SVC Educational Report",
            subtitle="Margin and support-vector explanation using sklearn SVC artifacts.",
            metadata={"model": "SVC", "accuracy": accuracy, "total_errors": errors},
            sections=sections,
        )


def _support_rows(model: SVC) -> TableRows:
    classes = getattr(model, "classes_", range(len(getattr(model, "n_support_", []))))
    return [
        {"class": str(class_label), "support_vectors": int(count)}
        for class_label, count in zip(classes, getattr(model, "n_support_", []), strict=True)
    ]


def _linear_coefficient_rows(model: SVC, names: list[str]) -> TableRows:
    if not hasattr(model, "coef_"):
        return []
    coefficients = np.asarray(model.coef_, dtype=float)
    if coefficients.ndim == 2:
        coefficients = coefficients[0]
    return feature_score_rows(names, coefficients, key="coefficient", absolute=False)


def _kernel_note(model: SVC) -> str:
    kernel = model.get_params().get("kernel")
    if kernel == "linear":
        return "A linear SVC exposes coefficients that can be inspected as linear score weights."
    return "For nonlinear kernels, SVC decisions happen in kernel space. Support vectors and scores are visible, but there is no simple feature-level rule."


def _linear_note(model: SVC) -> str:
    if hasattr(model, "coef_"):
        return "These coefficients are available because the SVC uses a linear kernel. They are not causal effects."
    return "No coefficients are shown because this SVC kernel does not expose a direct linear feature-weight vector."

