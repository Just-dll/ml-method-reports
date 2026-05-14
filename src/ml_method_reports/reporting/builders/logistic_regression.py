from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

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
    to_python,
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
class LogisticRegressionReportInput:
    model: LogisticRegression
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


class LogisticRegressionReportBuilder(ReportBuilder, ReportAssetBuilder):
    def __init__(self, report_input: LogisticRegressionReportInput, assets_dir: Path | None = None) -> None:
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
        names = feature_names(self._input.X_train, self._input.feature_names)
        coef_rows = _coefficient_rows(self._input.model, names)
        influence_rows = _influence_rows(self._input.model, names)
        assets: dict[str, str] = {}
        assets.update(build_bar_asset(coef_rows, assets_dir, "logistic_coefficients.png", label_key="feature", value_key="coefficient", title="Logistic Regression Coefficients"))
        assets.update(build_bar_asset(influence_rows, assets_dir, "logistic_feature_influence.png", label_key="feature", value_key="influence", title="Top Absolute Coefficients"))
        return assets

    def _build_report(self, assets: dict[str, str]) -> ExperimentReport:
        data = self._input
        X_test = as_2d_array(data.X_test)
        names = feature_names(data.X_train, data.feature_names)
        pred = predictions(data.model, data.X_test, data.predictions)
        index = selected_index(data.selected_sample_index, X_test.shape[0])
        query = selected_query(data.X_test, X_test, index)
        probs = probability_rows(data.model, query)
        scores = score_rows(data.model, query)
        coef_rows = _coefficient_rows(data.model, names)
        influence_rows = _influence_rows(data.model, names)
        confusion = confusion_rows(data.y_test, pred)
        errors = total_errors(data.y_test, pred)
        accuracy = float(np.mean(np.asarray(data.y_test) == pred))
        selected_prediction = pred[index]
        true_value = np.asarray(data.y_test)[index]

        sections = [
            ReportSection(title="1. Experiment Overview", table=[
                {"item": "dataset source", "value": data.dataset_source},
                {"item": "train samples", "value": int(as_2d_array(data.X_train).shape[0])},
                {"item": "test samples", "value": int(X_test.shape[0])},
                {"item": "features", "value": len(names)},
                {"item": "target", "value": data.target_name},
                {"item": "selected sample index", "value": index},
            ]),
            preprocessing_section(data.scaling_method, data.scaling_params, names),
            ReportSection(title="3. Model Parameters", table=model_params_rows(data.model, ["penalty", "C", "solver", "max_iter", "multi_class"])),
            ReportSection(title="4. Coefficients and Intercepts", content=_coefficient_note(), table=coef_rows, image_path=assets.get("logistic_coefficients")),
            ReportSection(title="5. Feature Influence", content="Influence is shown as absolute coefficient magnitude. This is predictive association, not causation.", table=influence_rows, image_path=assets.get("logistic_feature_influence")),
            ReportSection(title="6. Selected Prediction Explanation", content=_selected_text(index, true_value, selected_prediction), table=probs + scores),
            ReportSection(title="7. Evaluation", table=evaluation_rows(data.y_test, pred)),
            ReportSection(title="8. Confusion Matrix", table=confusion),
            ReportSection(title="9. Error Analysis", content=error_analysis_text(confusion)),
            ReportSection(title="10. Analysis Summary", content=f"LogisticRegression achieved {accuracy:.3f} accuracy with {errors} error(s). Coefficients describe linear score contributions and are scale-dependent, so preprocessing must be considered."),
        ]
        return ExperimentReport(
            title="Logistic Regression Educational Report",
            subtitle="Linear-model explanation using sklearn LogisticRegression artifacts.",
            metadata={"model": "LogisticRegression", "accuracy": accuracy, "total_errors": errors},
            sections=sections,
        )


def _coefficient_rows(model: LogisticRegression, names: list[str]) -> TableRows:
    coefficients = np.asarray(model.coef_, dtype=float)
    classes = getattr(model, "classes_", ["positive"])
    rows = []
    for class_index, values in enumerate(coefficients):
        class_label = classes[class_index] if len(classes) == coefficients.shape[0] else classes[-1]
        for name, value in zip(names, values, strict=True):
            rows.append({"class": to_python(class_label), "feature": name, "coefficient": float(value)})
    return sorted(rows, key=lambda row: abs(row["coefficient"]), reverse=True)


def _influence_rows(model: LogisticRegression, names: list[str]) -> TableRows:
    coefficients = np.asarray(model.coef_, dtype=float)
    if coefficients.ndim == 2:
        coefficients = np.abs(coefficients).mean(axis=0)
    return feature_score_rows(names, coefficients, key="influence", absolute=True)


def _coefficient_note() -> str:
    return "Signed coefficients show how features change linear class scores. They are not causal effects and depend on feature scaling."


def _selected_text(index: int, true_value: object, predicted: object) -> str:
    verdict = "correct" if predicted == true_value else "incorrect"
    return f"Selected sample {index} has true class {to_python(true_value)} and predicted class {to_python(predicted)}. The prediction is {verdict}."

