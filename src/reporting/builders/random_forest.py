from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    selected_index,
    selected_query,
    total_errors,
)
from ml_method_reports.reporting.models import ExperimentReport, ReportSection
from ml_method_reports.reporting.types import (
    FeatureMatrix,
    PredictionVector,
    ScalingParams,
    TargetVector,
)
from sklearn.ensemble import RandomForestClassifier


@dataclass(slots=True)
class RandomForestReportInput:
    model: RandomForestClassifier
    X_train: FeatureMatrix
    X_test: FeatureMatrix
    y_train: TargetVector
    y_test: TargetVector
    feature_names: list[str] | None = None
    predictions: PredictionVector | None = None
    dataset_source: str = "classification dataset"
    target_name: str = "target"
    selected_sample_index: int = 0
    preprocessing_method: str = "none"
    preprocessing_params: ScalingParams | None = None


class RandomForestReportBuilder(ReportBuilder, ReportAssetBuilder):
    def __init__(self, report_input: RandomForestReportInput, assets_dir: Path | None = None) -> None:
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
        importance = feature_score_rows(names, self._input.model.feature_importances_, key="importance")
        return build_bar_asset(importance, assets_dir, "random_forest_importance.png", label_key="feature", value_key="importance", title="Random Forest Feature Importance")

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
        importance = feature_score_rows(names, data.model.feature_importances_, key="importance")

        sections = [
            ReportSection(title="1. Experiment Overview", table=[
                {"item": "dataset source", "value": data.dataset_source},
                {"item": "train samples", "value": int(as_2d_array(data.X_train).shape[0])},
                {"item": "test samples", "value": int(X_test.shape[0])},
                {"item": "features", "value": len(names)},
                {"item": "selected sample index", "value": index},
            ]),
            preprocessing_section(data.preprocessing_method, data.preprocessing_params, names),
            ReportSection(title="3. Forest Parameters", table=model_params_rows(data.model, ["n_estimators", "criterion", "max_depth", "bootstrap", "random_state"])),
            ReportSection(title="4. Ensemble Concept", content="A RandomForest combines many decision trees. The final prediction is an aggregate vote/probability, not one global rule."),
            ReportSection(title="5. Feature Importance", content="Forest importance summarizes impurity reductions across trees. It is predictive importance, not causal importance.", table=importance, image_path=assets.get("random_forest_importance")),
            ReportSection(title="6. Selected Prediction Probabilities", table=probability_rows(data.model, query)),
            ReportSection(title="7. Example Tree Snapshot", content=_tree_snapshot_text(data.model)),
            ReportSection(title="8. Evaluation", table=evaluation_rows(data.y_test, pred)),
            ReportSection(title="9. Confusion Matrix", table=confusion),
            ReportSection(title="10. Error Analysis", content=error_analysis_text(confusion)),
            ReportSection(title="11. Analysis Summary", content=f"RandomForestClassifier achieved {accuracy:.3f} accuracy with {errors} error(s). This report explains forest-level artifacts while avoiding a false single-rule explanation."),
        ]
        return ExperimentReport(
            title="Random Forest Educational Report",
            subtitle="Ensemble explanation using sklearn RandomForestClassifier artifacts.",
            metadata={"model": "RandomForestClassifier", "accuracy": accuracy, "total_errors": errors},
            sections=sections,
        )


def _tree_snapshot_text(model: RandomForestClassifier) -> str:
    if not getattr(model, "estimators_", None):
        return "No fitted trees were found on the model."
    first_tree = model.estimators_[0]
    return (
        f"The first tree has depth {first_tree.get_depth()} and "
        f"{first_tree.get_n_leaves()} leaves. This tree is one member of the ensemble, "
        "not the full forest explanation."
    )

