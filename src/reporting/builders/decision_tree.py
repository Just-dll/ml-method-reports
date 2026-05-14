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
    TableRows,
    TargetVector,
)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


@dataclass(slots=True)
class DecisionTreeReportInput:
    model: DecisionTreeClassifier
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


class DecisionTreeReportBuilder(ReportBuilder, ReportAssetBuilder):
    def __init__(self, report_input: DecisionTreeReportInput, assets_dir: Path | None = None) -> None:
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
        importance = feature_score_rows(names, getattr(self._input.model, "feature_importances_", []), key="importance", absolute=False)
        assets: dict[str, str] = {}
        assets.update(build_bar_asset(importance, assets_dir, "decision_tree_importance.png", label_key="feature", value_key="importance", title="Decision Tree Feature Importance"))
        assets.update(_plot_tree_asset(self._input.model, assets_dir, names))
        return assets

    def _build_report(self, assets: dict[str, str]) -> ExperimentReport:
        data = self._input
        X_test = as_2d_array(data.X_test)
        names = feature_names(data.X_train, data.feature_names)
        pred = predictions(data.model, data.X_test, data.predictions)
        index = selected_index(data.selected_sample_index, X_test.shape[0])
        query = selected_query(data.X_test, X_test, index)
        path_rows = _decision_path_rows(data.model, query, names)
        leaf_id = int(data.model.apply(query)[0])
        confusion = confusion_rows(data.y_test, pred)
        errors = total_errors(data.y_test, pred)
        accuracy = float(np.mean(np.asarray(data.y_test) == pred))

        sections = [
            ReportSection(title="1. Experiment Overview", table=[
                {"item": "dataset source", "value": data.dataset_source},
                {"item": "train samples", "value": int(as_2d_array(data.X_train).shape[0])},
                {"item": "test samples", "value": int(X_test.shape[0])},
                {"item": "features", "value": len(names)},
                {"item": "selected sample index", "value": index},
            ]),
            preprocessing_section(data.preprocessing_method, data.preprocessing_params, names),
            ReportSection(title="3. Tree Parameters", table=model_params_rows(data.model, ["criterion", "max_depth", "min_samples_split", "min_samples_leaf", "random_state"])),
            ReportSection(title="4. Tree Structure Summary", content="A tree predicts by following split rules from root to leaf.", table=[
                {"metric": "depth", "value": data.model.get_depth()},
                {"metric": "leaves", "value": data.model.get_n_leaves()},
                {"metric": "node_count", "value": int(data.model.tree_.node_count)},
            ], image_path=assets.get("decision_tree")),
            ReportSection(title="5. Feature Importance", content="Impurity-based importance is useful but can be biased toward high-cardinality features.", table=feature_score_rows(names, data.model.feature_importances_, key="importance"), image_path=assets.get("decision_tree_importance")),
            ReportSection(title="6. Selected Sample Decision Path", content=f"Selected sample {index} ends in leaf {leaf_id}. A path explains this model decision, not causality.", table=path_rows),
            ReportSection(title="7. Leaf Prediction Details", table=probability_rows(data.model, query)),
            ReportSection(title="8. Evaluation", table=evaluation_rows(data.y_test, pred)),
            ReportSection(title="9. Confusion Matrix", table=confusion),
            ReportSection(title="10. Error Analysis", content=error_analysis_text(confusion)),
            ReportSection(title="11. Analysis Summary", content=f"DecisionTreeClassifier achieved {accuracy:.3f} accuracy with {errors} error(s). The report exposes rules and a selected-sample path, while warning that tree importances are not causal."),
            ReportSection(title="12. Text Rules", code=export_text(data.model, feature_names=names, max_depth=4)),
        ]
        return ExperimentReport(
            title="Decision Tree Educational Report",
            subtitle="Rule-path explanation using sklearn DecisionTreeClassifier artifacts.",
            metadata={"model": "DecisionTreeClassifier", "accuracy": accuracy, "total_errors": errors},
            sections=sections,
        )


def _decision_path_rows(model: DecisionTreeClassifier, query: FeatureMatrix, names: list[str]) -> TableRows:
    node_indicator = model.decision_path(query)
    leaf_id = int(model.apply(query)[0])
    rows = []
    for node_id in node_indicator.indices[node_indicator.indptr[0]: node_indicator.indptr[1]]:
        if node_id == leaf_id:
            rows.append({"node": int(node_id), "rule": "leaf", "value": "prediction made here"})
            continue
        feature_index = int(model.tree_.feature[node_id])
        threshold = float(model.tree_.threshold[node_id])
        rows.append({"node": int(node_id), "feature": names[feature_index], "threshold": threshold})
    return rows


def _plot_tree_asset(model: DecisionTreeClassifier, assets_dir: Path, names: list[str]) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = assets_dir / "decision_tree.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    plot_tree(model, feature_names=names, class_names=[str(value) for value in getattr(model, "classes_", [])], max_depth=3, filled=True, fontsize=7, ax=ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"decision_tree": str(Path("assets") / path.name)}

