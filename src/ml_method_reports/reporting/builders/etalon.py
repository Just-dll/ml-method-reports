from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml_method_reports.algorithms import (
    EtalonClassifier,
    EtalonEvaluationSummary,
    EtalonExplanation,
)
from ml_method_reports.reporting.builders.base import ReportAssetBuilder, ReportBuilder
from ml_method_reports.reporting.models import ExperimentReport, ReportSection
from ml_method_reports.reporting.plots import (
    plot_etalon_decision_space,
    plot_full_vs_one_feature_comparison,
    plot_model_accuracy_comparison,
    project_to_2d,
)
from ml_method_reports.reporting.types import TableRows


@dataclass(slots=True)
class EtalonRun:
    mode: str
    model: EtalonClassifier
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    predictions: np.ndarray
    feature_names: list[str]
    summary: EtalonEvaluationSummary
    test_indices: np.ndarray


@dataclass(slots=True)
class EtalonReportInput:
    dataframe: pd.DataFrame
    train_indices: np.ndarray
    test_indices: np.ndarray
    full_run: EtalonRun
    one_feature_run: EtalonRun
    comparison_rows: TableRows
    metric: str
    normalization: str
    prototype_strategy: str
    feature_index: int
    one_feature_name: str
    random_state: int
    dataset_source: str = "synthetic classification dataset"
    target_name: str = "target"


class EtalonReportBuilder(ReportBuilder, ReportAssetBuilder):
    def __init__(
        self,
        report_input: EtalonReportInput,
        assets_dir: Path | None = None,
    ) -> None:
        self._input = report_input
        self._assets_dir = assets_dir

    def build(self) -> ExperimentReport:
        visual_assets = self.build_assets(self._assets_dir) if self._assets_dir else {}
        return self._build_report(visual_assets)

    def build_assets(self, assets_dir: Path | None = None) -> dict[str, str]:
        if assets_dir is None:
            if self._assets_dir is None:
                raise ValueError("assets_dir must be provided to build report assets.")
            assets_dir = self._assets_dir
        return _build_visual_assets(
            full_run=self._input.full_run,
            one_feature_run=self._input.one_feature_run,
            comparison_rows=self._input.comparison_rows,
            assets_dir=assets_dir,
        )

    def _build_report(self, visual_assets: dict[str, str]) -> ExperimentReport:
        data = self._input
        explanation = data.full_run.model.explain_prediction(
            data.full_run.X_test,
            y_true=data.full_run.y_test,
            sample_index=0,
        )
        one_feature_explanation = data.one_feature_run.model.explain_prediction(
            data.one_feature_run.X_test,
            y_true=data.one_feature_run.y_test,
            sample_index=0,
        )
        full_vs_one_rows = [
            comparison_summary_row(data.full_run),
            comparison_summary_row(data.one_feature_run),
        ]
        train_distribution = class_distribution_rows(data.full_run.y_train, data.full_run.y_test)
        center_note = center_space_note(data.normalization)
        analysis = build_analysis_summary(
            full_run=data.full_run,
            one_feature_run=data.one_feature_run,
            comparison_rows=data.comparison_rows,
        )
        feature_count = len(data.full_run.feature_names)
        classes = sorted(data.dataframe[data.target_name].unique().tolist())

        sections = [
            ReportSection(
                title="1. Experiment Overview",
                table=[
                    {"item": "dataset source", "value": data.dataset_source},
                    {"item": "samples", "value": len(data.dataframe)},
                    {"item": "train samples", "value": len(data.train_indices)},
                    {"item": "test samples", "value": len(data.test_indices)},
                    {"item": "test size", "value": len(data.test_indices) / len(data.dataframe)},
                    {"item": "features", "value": feature_count},
                    {"item": "classes", "value": classes},
                    {"item": "metric", "value": data.metric},
                    {"item": "normalization", "value": data.normalization},
                    {"item": "prototype_strategy", "value": data.prototype_strategy},
                    {"item": "selected one-feature index", "value": data.feature_index},
                    {"item": "selected one-feature name", "value": data.one_feature_name},
                    {"item": "random_state", "value": data.random_state},
                ],
            ),
            ReportSection(
                title="1b. Train/Test Split",
                content=(
                    "Evaluation metrics are computed on the held-out test split, "
                    "not on all rows."
                ),
                table=train_distribution,
            ),
            ReportSection(
                title="2. Method Description",
                content=(
                    "The method of etalons is a supervised metric classification method. "
                    "For each class, it builds a prototype/etalon, usually the class centroid. "
                    "A new sample is assigned to the class of the nearest etalon according to "
                    "a selected distance metric."
                ),
            ),
            ReportSection(
                title="2b. Etalon Decision Visualization",
                content=(
                    "Points are test samples. X markers are learned class etalons. "
                    "The star is the selected sample, and dashed lines show distances "
                    "from that sample to each etalon."
                ),
                image_path=visual_assets.get("decision"),
                image_caption=(
                    "Decision space view: "
                    f"{visual_assets.get('projection_info', 'not generated')}. "
                    "Red-edged points indicate recognition errors."
                ),
            ),
            ReportSection(
                title="3. Dataset Preview",
                table=data.dataframe.head(8).to_dict(orient="records"),
            ),
            ReportSection(
                title="4. Normalization Details",
                content=normalization_description(data.normalization),
                table=normalization_rows(data.full_run.model, data.full_run.feature_names),
            ),
            ReportSection(
                title="5. Learned Etalons / Class Centers",
                content=center_note,
                table=center_rows(data.full_run),
            ),
            ReportSection(
                title="5b. One-feature Learned Etalons",
                content=center_note,
                table=center_rows(data.one_feature_run),
            ),
            ReportSection(
                title="6. Prediction Explanation",
                table=explanation_summary_rows(explanation),
            ),
            ReportSection(
                title="6b. Distances To Class Etalons",
                table=distance_rows(explanation),
            ),
            ReportSection(
                title="6c. One-feature Prediction Explanation",
                table=explanation_summary_rows(one_feature_explanation),
            ),
            ReportSection(
                title="7. Full-feature Evaluation",
                table=evaluation_rows(data.full_run.summary),
            ),
            ReportSection(
                title="8. One-feature Evaluation",
                table=evaluation_rows(data.one_feature_run.summary),
            ),
            ReportSection(
                title="9. Full vs One-feature Comparison",
                table=full_vs_one_rows,
            ),
            ReportSection(
                title="9b. Full-feature vs One-feature Visualization",
                image_path=visual_assets.get("full_vs_one"),
                image_caption=(
                    "Accuracy comparison for the full feature set and selected "
                    "one-feature run."
                ),
            ),
            ReportSection(title="10. Sklearn Comparison", table=data.comparison_rows),
            ReportSection(
                title="10b. Model Accuracy Comparison",
                image_path=visual_assets.get("accuracy"),
                image_caption="EtalonClassifier is marked as the custom educational method.",
            ),
            ReportSection(title="11. Analysis Summary", content=analysis),
        ]
        return ExperimentReport(
            title="Method of Etalons Classification Protocol",
            subtitle="Educational metric-classification experiment with prototype explanations.",
            metadata={
                "dataset_source": data.dataset_source,
                "samples": len(data.dataframe),
                "features": feature_count,
                "classes": classes,
                "metric": data.metric,
                "normalization": data.normalization,
                "prototype_strategy": data.prototype_strategy,
                "selected_feature_index": data.feature_index,
            },
            sections=sections,
        )


def normalization_description(normalization: str) -> str:
    if normalization == "none":
        return "Normalization disabled."
    if normalization == "standard":
        return "Standard normalization uses X_norm = (X - mean) / std."
    if normalization == "minmax":
        return "Min-max normalization uses X_norm = (X - min) / range."
    if normalization == "maxabs":
        return "Max-abs normalization uses X_norm = X / max(abs(X))."
    if normalization == "robust":
        return "Robust normalization uses X_norm = (X - median) / IQR."
    return f"Normalization mode {normalization!r}."


def normalization_rows(model: EtalonClassifier, feature_names: list[str]) -> TableRows:
    if model.normalization_ == "none":
        return []
    if model.normalization_ == "standard":
        return [
            {"feature": name, "mean": mean, "std": std}
            for name, mean, std in zip(
                feature_names,
                model.normalization_params_["mean"],
                model.normalization_params_["std"],
                strict=True,
            )
        ]
    if model.normalization_ == "minmax":
        return [
            {"feature": name, "min": minimum, "range": value_range}
            for name, minimum, value_range in zip(
                feature_names,
                model.normalization_params_["min"],
                model.normalization_params_["range"],
                strict=True,
            )
        ]
    if model.normalization_ == "maxabs":
        return [
            {"feature": name, "max_abs": max_abs}
            for name, max_abs in zip(
                feature_names,
                model.normalization_params_["max_abs"],
                strict=True,
            )
        ]
    if model.normalization_ == "robust":
        return [
            {"feature": name, "median": median, "iqr": iqr}
            for name, median, iqr in zip(
                feature_names,
                model.normalization_params_["median"],
                model.normalization_params_["iqr"],
                strict=True,
            )
        ]
    return []


def center_space_note(normalization: str) -> str:
    if normalization == "none":
        return "Centers are shown in the original feature space."
    if normalization == "minmax":
        return "Centers are shown in min-max normalized feature space."
    if normalization == "standard":
        return "Centers are shown in the normalized feature space because normalization='standard'."
    if normalization == "maxabs":
        return "Centers are shown in max-abs normalized feature space."
    if normalization == "robust":
        return "Centers are shown in robust normalized feature space."
    return "Centers are shown in the normalized feature space."


def center_rows(run: EtalonRun) -> TableRows:
    centers = run.model.get_class_centers()
    rows = []
    for class_label, center in centers.items():
        sample_count = int(np.count_nonzero(run.y_train == class_label))
        for feature_name, value in zip(run.feature_names, center, strict=True):
            rows.append(
                {
                    "class": class_label,
                    "sample_count": sample_count,
                    "feature": feature_name,
                    "center_value": value,
                }
            )
    return rows


def class_distribution_rows(y_train: np.ndarray, y_test: np.ndarray) -> TableRows:
    labels = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())
    return [
        {
            "class": label,
            "train_count": int(np.count_nonzero(y_train == label)),
            "test_count": int(np.count_nonzero(y_test == label)),
        }
        for label in labels
    ]


def explanation_summary_rows(explanation: EtalonExplanation) -> TableRows:
    return [
        {
            "sample_index": explanation.sample_index,
            "true_class": explanation.true_class,
            "predicted_class": explanation.predicted_class,
            "nearest_etalon_index": explanation.nearest_etalon_index,
            "nearest_etalon_class": explanation.nearest_etalon_class,
            "decision": "correct" if explanation.is_correct else "incorrect",
        }
    ]


def distance_rows(explanation: EtalonExplanation) -> TableRows:
    return [
        {"class": class_label, "distance": distance}
        for class_label, distance in explanation.distances.items()
    ]


def evaluation_rows(summary: EtalonEvaluationSummary) -> TableRows:
    return [
        {"metric": "total samples", "value": summary.total_samples},
        {"metric": "correct count", "value": summary.correct_count},
        {"metric": "error count", "value": summary.error_count},
        {"metric": "accuracy", "value": summary.accuracy},
        {"metric": "error rate", "value": summary.error_rate},
        {"metric": "P(correct)", "value": summary.probability_correct},
        {"metric": "P(error)", "value": summary.probability_error},
        {"metric": "classes", "value": summary.classes},
        {"metric": "confusion matrix", "value": summary.confusion_matrix},
    ]


def comparison_summary_row(run: EtalonRun) -> dict[str, object]:
    return {
        "mode": run.mode,
        "feature_count": len(run.feature_names),
        "accuracy": run.summary.accuracy,
        "error_rate": run.summary.error_rate,
        "P(correct)": run.summary.probability_correct,
        "P(error)": run.summary.probability_error,
    }


def build_analysis_summary(
    *,
    full_run: EtalonRun,
    one_feature_run: EtalonRun,
    comparison_rows: TableRows,
) -> str:
    full_accuracy = full_run.summary.accuracy
    one_accuracy = one_feature_run.summary.accuracy
    sklearn_rows = [row for row in comparison_rows if row.get("type") == "sklearn"]
    if full_accuracy > one_accuracy + 0.02:
        feature_message = (
            "The full-feature experiment performed better, suggesting that multiple "
            "features improve class separability on this split."
        )
    elif one_accuracy > full_accuracy + 0.02:
        feature_message = (
            "The selected feature performed better on this split, which can happen "
            "when one synthetic signal is especially informative."
        )
    else:
        feature_message = (
            "The full-feature and one-feature results are similar, although their "
            "confusion matrices may still differ."
        )

    baseline_message = ""
    if sklearn_rows:
        best_baseline = max(sklearn_rows, key=lambda row: row["accuracy"])
        baseline_message = (
            f"The best sklearn baseline was {best_baseline['model']} with "
            f"{best_baseline['accuracy']:.3f} accuracy. "
        )

    return (
        f"The EtalonClassifier achieved {full_accuracy:.3f} accuracy with "
        f"{full_run.summary.error_count} recognition errors. "
        f"The one-feature experiment achieved {one_accuracy:.3f} accuracy. "
        f"{feature_message} "
        f"{baseline_message}"
        "EtalonClassifier is an educational prototype-based method, not necessarily "
        "the strongest production model."
    )


def _build_visual_assets(
    *,
    full_run: EtalonRun,
    one_feature_run: EtalonRun,
    comparison_rows: TableRows,
    assets_dir: Path,
) -> dict[str, str]:
    normalized_test = _transform_for_visualization(full_run.model, full_run.X_test)
    centers = np.vstack(list(full_run.model.get_class_centers().values()))
    X_2d, centers_2d, projection_info = project_to_2d(normalized_test, centers)

    decision_path = plot_etalon_decision_space(
        X_2d=X_2d,
        y_true=full_run.y_test,
        y_pred=full_run.predictions,
        centers_2d=centers_2d,
        selected_sample_index=0,
        class_labels=full_run.model.classes_,
        output_path=assets_dir / "etalon_decision_space.png",
        title=f"Etalon Decision Space ({projection_info})",
    )
    accuracy_path = plot_model_accuracy_comparison(
        comparison_rows=comparison_rows,
        output_path=assets_dir / "model_accuracy_comparison.png",
    )
    full_vs_one_path = plot_full_vs_one_feature_comparison(
        comparison_rows=[
            comparison_summary_row(full_run),
            comparison_summary_row(one_feature_run),
        ],
        output_path=assets_dir / "full_vs_one_feature.png",
    )
    return {
        "decision": str(Path("assets") / decision_path.name),
        "accuracy": str(Path("assets") / accuracy_path.name),
        "full_vs_one": str(Path("assets") / full_vs_one_path.name),
        "projection_info": projection_info,
    }


def _transform_for_visualization(model: EtalonClassifier, X: np.ndarray) -> np.ndarray:
    if model.normalization_ == "none":
        return X
    if model.normalization_ == "standard":
        return (X - model.normalization_params_["mean"]) / model.normalization_params_["std"]
    if model.normalization_ == "minmax":
        return (X - model.normalization_params_["min"]) / model.normalization_params_["range"]
    if model.normalization_ == "maxabs":
        return X / model.normalization_params_["max_abs"]
    if model.normalization_ == "robust":
        return (X - model.normalization_params_["median"]) / model.normalization_params_["iqr"]
    raise RuntimeError(f"Unsupported normalization {model.normalization_!r}.")

