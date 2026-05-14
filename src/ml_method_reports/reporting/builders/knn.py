from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier

from ml_method_reports.reporting.builders.base import ReportAssetBuilder, ReportBuilder
from ml_method_reports.reporting.models import ExperimentReport, ReportSection
from ml_method_reports.reporting.plots import plot_knn_neighbor_space, project_to_2d
from ml_method_reports.reporting.types import (
    FeatureMatrix,
    PredictionVector,
    ScalingParams,
    TableRow,
    TableRows,
    TargetVector,
)


@dataclass(slots=True)
class KnnReportInput:
    model: KNeighborsClassifier
    X_train: FeatureMatrix
    X_test: FeatureMatrix
    y_train: TargetVector
    y_test: TargetVector
    feature_names: list[str] | None = None
    predictions: PredictionVector | None = None
    leaderboard_rows: TableRows | None = None
    dataset_source: str = "classification dataset"
    target_name: str = "target"
    selected_sample_index: int = 0
    selected_sample_indices: list[int] | None = None
    scaling_method: str = "none"
    scaling_params: ScalingParams | None = None


class KnnReportBuilder(ReportBuilder, ReportAssetBuilder):
    def __init__(
        self,
        report_input: KnnReportInput,
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
        return _build_visual_assets(self._input, assets_dir)

    def _build_report(self, visual_assets: dict[str, str]) -> ExperimentReport:
        data = self._input
        _validate_knn_model(data.model)
        X_train = _as_2d_array(data.X_train)
        X_test = _as_2d_array(data.X_test)
        y_train = np.asarray(data.y_train)
        y_test = np.asarray(data.y_test)
        _validate_data_shapes(X_train, X_test, y_train, y_test)

        feature_names = _feature_names(data.X_train, data.feature_names, X_train.shape[1])
        predictions = _predictions(data.model, data.X_test, X_test, data.predictions)
        selected_indices = _selected_indices(data, y_test, X_test.shape[0])
        selected_cases = _selected_cases(data, y_test, predictions, selected_indices)
        metrics = _evaluation_rows(y_test, predictions)
        confusion_rows = _confusion_rows(y_test, predictions)
        accuracy = float(accuracy_score(y_test, predictions))
        total_errors = int(np.count_nonzero(y_test != predictions))
        params = data.model.get_params()
        classes = getattr(data.model, "classes_", np.unique(np.concatenate([y_train, y_test]))).tolist()
        decision_text = _decision_explanation(selected_cases, params.get("weights", "uniform"))

        sections = [
            ReportSection(
                title="1. Experiment Overview",
                table=[
                    {"item": "dataset source", "value": data.dataset_source},
                    {"item": "train samples", "value": int(X_train.shape[0])},
                    {"item": "test samples", "value": int(X_test.shape[0])},
                    {"item": "selected samples", "value": len(selected_cases)},
                    {"item": "features", "value": int(X_train.shape[1])},
                    {"item": "classes", "value": classes},
                    {"item": "target", "value": data.target_name},
                    {"item": "selected sample indices", "value": selected_indices},
                ],
            ),
            ReportSection(
                title="2. KNN Method Description",
                content=(
                    "KNeighborsClassifier assigns a class by looking at the nearest "
                    "training samples under the configured distance metric and then "
                    "aggregating their votes."
                ),
            ),
            ReportSection(title="3. Model Parameters", table=_parameter_rows(params)),
            ReportSection(title="4. Feature Columns", table=_feature_rows(feature_names)),
            ReportSection(
                title="5. Preprocessing / Normalization Summary",
                content=_preprocessing_summary(data.scaling_method),
                table=_scaling_rows(
                    scaling_method=data.scaling_method,
                    scaling_params=data.scaling_params,
                    feature_names=feature_names,
                ),
            ),
            ReportSection(
                title="6. Nearest Neighbor Visualization",
                content=_visualization_note(visual_assets.get("projection_info"), len(selected_cases)),
                image_path=visual_assets.get("neighbors"),
                image_caption=(
                    "Training samples are projected to two dimensions. "
                    "Selected test samples and their nearest training neighbors are highlighted. "
                    f"{visual_assets.get('projection_info', 'Visualization was not generated')}."
                ),
            ),
            ReportSection(
                title="7. KNN vs Method of Etalons",
                content=(
                    "EtalonClassifier compares a sample to learned class prototypes or centers. "
                    "KNN compares a sample to individual training objects. Both are metric "
                    "methods: the decision comes from distances in feature space."
                ),
            ),
            ReportSection(title="8. Selected Predictions", table=_selected_prediction_rows(selected_cases)),
            ReportSection(title="9. Nearest Neighbors", table=_combined_neighbor_rows(selected_cases)),
            ReportSection(title="10. Neighbor Votes", table=_combined_vote_rows(selected_cases)),
            ReportSection(
                title="11. Decision Explanation",
                content=decision_text,
            ),
            ReportSection(title="12. Evaluation", table=metrics),
            ReportSection(
                title="13. Confusion Matrix",
                table=confusion_rows,
            ),
            ReportSection(
                title="14. Error Analysis",
                content=_error_analysis_text(confusion_rows, total_errors),
                table=_misclassification_rows(confusion_rows),
            ),
            ReportSection(
                title="15. Model Comparison",
                table=data.leaderboard_rows or [],
            ),
            ReportSection(
                title="16. Analysis Summary",
                content=_analysis_summary(
                    accuracy=accuracy,
                    total_errors=total_errors,
                    decision_text=decision_text,
                ),
            ),
        ]
        return ExperimentReport(
            title="KNN Classification Report",
            subtitle="Nearest-neighbor explanation generated from sklearn KNeighborsClassifier.",
            metadata={
                "dataset_source": data.dataset_source,
                "model": "KNeighborsClassifier",
                "n_neighbors": params.get("n_neighbors"),
                "weights": params.get("weights"),
                "metric": params.get("metric"),
                "algorithm": params.get("algorithm"),
                "accuracy": accuracy,
                "total_errors": total_errors,
                "scaling_method": data.scaling_method,
            },
            sections=sections,
        )


def _preprocessing_summary(scaling_method: str) -> str:
    if scaling_method == "standard":
        return (
            "KNN is distance-based, so feature scaling matters. This report uses "
            "standard scaling before neighbor search: each feature is centered by "
            "its training mean and divided by its training standard deviation."
        )
    if scaling_method == "none":
        return (
            "No explicit scaling metadata was provided. For KNN, unscaled features "
            "with larger numeric ranges can dominate neighbor distances."
        )
    return (
        f"The KNN input uses {scaling_method} preprocessing before neighbor search. "
        "Because KNN is distance-based, preprocessing directly affects nearest neighbors."
    )


def _scaling_rows(
    *,
    scaling_method: str,
    scaling_params: ScalingParams | None,
    feature_names: list[str],
) -> TableRows:
    if scaling_method != "standard" or not scaling_params:
        return [{"scaling": scaling_method, "details": "exact scaling parameters not provided"}]

    means = scaling_params.get("mean")
    scales = scaling_params.get("scale")
    if scales is None:
        scales = scaling_params.get("std")
    if means is None or scales is None:
        return [{"scaling": "standard", "details": "StandardScaler parameters not provided"}]

    return [
        {"feature": feature_name, "mean": float(mean), "std": float(scale)}
        for feature_name, mean, scale in zip(feature_names, means, scales, strict=True)
    ]


def _visualization_note(projection_info: str | None, selected_count: int) -> str:
    if projection_info and "PCA" in projection_info:
        return (
            "This plot is a 2D PCA projection for visualization only. The actual "
            "nearest-neighbor distances shown in the tables are computed in the "
            "original/preprocessed feature space, not from visual spacing on the plot. "
            f"{selected_count} selected sample(s) are highlighted."
        )
    return (
        "The plot is a compact 2D view of the neighbor relationship. Numeric neighbor "
        "distances in the tables are computed in the original/preprocessed feature space. "
        f"{selected_count} selected sample(s) are highlighted."
    )


def _decision_explanation(selected_cases: list[dict[str, object]], weights: object) -> str:
    if not selected_cases:
        return "No selected samples were available for explanation."
    voting_rule = (
        "distance-weighted voting"
        if weights == "distance"
        else "majority voting with uniform weights"
    )
    parts = []
    for case in selected_cases:
        vote_text = ", ".join(
            f"class {row['class']}: {row['neighbor_count']} neighbor(s)"
            for row in case["vote_rows"]
        )
        winning_vote = case["vote_rows"][0] if case["vote_rows"] else {"class": case["explanation"]["predicted_class"]}
        correctness = "correct" if case["explanation"]["is_correct"] else "incorrect"
        parts.append(
            f"Sample {case['explanation']['sample_index']} has true class {case['explanation']['true_class']}, "
            f"predicted class {case['explanation']['predicted_class']}, and nearest-neighbor votes: {vote_text}. "
            f"With {voting_rule}, class {winning_vote['class']} leads the decision. "
            f"The prediction is {correctness}."
        )
    return " ".join(parts)


def _selected_indices(data: KnnReportInput, y_test: np.ndarray, sample_count: int) -> list[int]:
    indices = _default_selected_indices(y_test, sample_count)
    if not indices:
        indices = [_selected_index(data.selected_sample_index, sample_count)]
    if data.selected_sample_indices is not None:
        for index in data.selected_sample_indices:
            candidate = _selected_index(int(index), sample_count)
            if candidate not in indices:
                indices.append(candidate)
    return indices


def _default_selected_indices(y_test: np.ndarray, sample_count: int) -> list[int]:
    seen: set[object] = set()
    indices: list[int] = []
    for index, label in enumerate(y_test):
        if label in seen:
            continue
        seen.add(label)
        indices.append(index)
        if len(indices) >= min(len(np.unique(y_test)), sample_count):
            break
    return indices


def _selected_cases(
    data: KnnReportInput,
    y_test: np.ndarray,
    predictions: np.ndarray,
    selected_indices: list[int],
) -> list[dict[str, object]]:
    cases = []
    for selected_index in selected_indices:
        selected_query = _selected_query(data.X_test, _as_2d_array(data.X_test), selected_index)
        distances, indices = data.model.kneighbors(selected_query)
        explanation = _prediction_explanation(
            model=data.model,
            selected_query=selected_query,
            y_test=y_test,
            predictions=predictions,
            selected_index=selected_index,
        )
        neighbor_rows = _neighbor_rows(
            y_train=np.asarray(data.y_train),
            distances=distances[0],
            indices=indices[0],
            weights=data.model.get_params().get("weights", "uniform"),
        )
        cases.append(
            {
                "explanation": explanation,
                "neighbor_rows": neighbor_rows,
                "vote_rows": _vote_rows(neighbor_rows),
            }
        )
    return cases


def _selected_prediction_rows(selected_cases: list[dict[str, object]]) -> TableRows:
    rows: TableRows = []
    for case in selected_cases:
        explanation = case["explanation"]
        row = dict(explanation)
        row["vote_summary"] = ", ".join(
            f"class {vote['class']}: {vote['neighbor_count']} neighbor(s)"
            for vote in case["vote_rows"]
        )
        rows.append(row)
    return rows


def _combined_neighbor_rows(selected_cases: list[dict[str, object]]) -> TableRows:
    rows: TableRows = []
    for case in selected_cases:
        sample_index = case["explanation"]["sample_index"]
        for row in case["neighbor_rows"]:
            rows.append({"sample_index": sample_index, **row})
    return rows


def _combined_vote_rows(selected_cases: list[dict[str, object]]) -> TableRows:
    rows: TableRows = []
    for case in selected_cases:
        sample_index = case["explanation"]["sample_index"]
        for row in case["vote_rows"]:
            rows.append({"sample_index": sample_index, **row})
    return rows


def _error_analysis_text(confusion_rows: TableRows, total_errors: int) -> str:
    if total_errors == 0:
        return "There are 0 recognition errors on the test split."
    mistakes = _misclassification_rows(confusion_rows)
    mistake_text = "; ".join(
        f"class {row['true_class']} predicted as {row['predicted_class']}: {row['count']}"
        for row in mistakes
    )
    return f"There are {total_errors} recognition error(s) on the test split. {mistake_text}."


def _misclassification_rows(confusion_rows: TableRows) -> TableRows:
    return [
        row
        for row in confusion_rows
        if row["true_class"] != row["predicted_class"] and row["count"] > 0
    ]


def _analysis_summary(*, accuracy: float, total_errors: int, decision_text: str) -> str:
    return (
        f"The KNN model achieved {accuracy:.3f} accuracy with {total_errors} total "
        f"recognition error(s). {decision_text} This report uses the sklearn "
        "KNeighborsClassifier implementation and exposes educational artifacts: nearest "
        "neighbors, distances, and voting."
    )


def _build_visual_assets(data: KnnReportInput, assets_dir: Path) -> dict[str, str]:
    X_train = _as_2d_array(data.X_train)
    X_test = _as_2d_array(data.X_test)
    selected_index = _selected_index(data.selected_sample_index, X_test.shape[0])
    distances, indices = data.model.kneighbors(_selected_query(data.X_test, X_test, selected_index))
    combined = np.vstack([X_train, X_test[[selected_index]]])
    combined_2d, _, projection_info = project_to_2d(combined)
    train_2d = combined_2d[: X_train.shape[0]]
    selected_2d = combined_2d[X_train.shape[0]]
    path = plot_knn_neighbor_space(
        X_train_2d=train_2d,
        y_train=data.y_train,
        selected_sample_2d=selected_2d,
        neighbor_indices=indices[0],
        neighbor_distances=distances[0],
        output_path=assets_dir / "knn_neighbor_space.png",
    )
    return {
        "neighbors": str(Path("assets") / path.name),
        "projection_info": projection_info,
    }


def _validate_knn_model(model: object) -> None:
    if not hasattr(model, "kneighbors"):
        raise TypeError("KnnReportBuilder requires a fitted sklearn KNeighborsClassifier-like model.")


def _validate_data_shapes(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be 2D arrays.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test must have the same number of features.")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("y_train must contain one value per training sample.")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError("y_test must contain one value per test sample.")


def _as_2d_array(value: FeatureMatrix) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _feature_names(X_train: FeatureMatrix, provided: list[str] | None, feature_count: int) -> list[str]:
    if provided is not None:
        return provided
    if isinstance(X_train, pd.DataFrame):
        return X_train.columns.astype(str).tolist()
    return [f"feature_{index}" for index in range(feature_count)]


def _predictions(
    model: KNeighborsClassifier,
    original_X_test: FeatureMatrix,
    X_test: np.ndarray,
    predictions: PredictionVector | None,
) -> np.ndarray:
    if predictions is None:
        return np.asarray(model.predict(original_X_test))
    predicted_values = np.asarray(predictions)
    if predicted_values.shape[0] != X_test.shape[0]:
        raise ValueError("predictions must contain one value per test sample.")
    return predicted_values


def _selected_index(selected_index: int, sample_count: int) -> int:
    if selected_index < 0 or selected_index >= sample_count:
        raise ValueError(
            f"selected_sample_index must be between 0 and {sample_count - 1}; got {selected_index}."
        )
    return int(selected_index)


def _selected_query(original_X_test: FeatureMatrix, X_test: np.ndarray, selected_index: int) -> FeatureMatrix:
    if isinstance(original_X_test, pd.DataFrame):
        return original_X_test.iloc[[selected_index]]
    return X_test[[selected_index]]


def _prediction_explanation(
    *,
    model: KNeighborsClassifier,
    selected_query: FeatureMatrix,
    y_test: np.ndarray,
    predictions: np.ndarray,
    selected_index: int,
) -> TableRow:
    predicted_class = predictions[selected_index]
    row = {
        "sample_index": selected_index,
        "true_class": _to_python_value(y_test[selected_index]),
        "predicted_class": _to_python_value(predicted_class),
        "is_correct": bool(predicted_class == y_test[selected_index]),
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(selected_query)[0]
        classes = getattr(model, "classes_", [])
        for class_label, probability in zip(classes, probabilities, strict=True):
            row[f"probability_{class_label}"] = float(probability)
    return row


def _neighbor_rows(
    *,
    y_train: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    weights: object,
) -> TableRows:
    vote_weights = _vote_weights(distances, weights)
    return [
        {
            "rank": rank,
            "train_index": int(train_index),
            "neighbor_class": _to_python_value(y_train[train_index]),
            "distance": float(distance),
            "vote_weight": float(vote_weight),
        }
        for rank, (train_index, distance, vote_weight) in enumerate(
            zip(indices, distances, vote_weights, strict=True),
            start=1,
        )
    ]


def _vote_weights(distances: np.ndarray, weights: object) -> np.ndarray:
    if weights == "distance":
        safe_distances = np.where(distances < 1e-12, 1e-12, distances)
        return 1.0 / safe_distances
    return np.ones_like(distances, dtype=float)


def _vote_rows(neighbor_rows: TableRows) -> TableRows:
    votes: dict[object, TableRow] = {}
    for row in neighbor_rows:
        class_label = row["neighbor_class"]
        if class_label not in votes:
            votes[class_label] = {"class": class_label, "neighbor_count": 0, "vote_weight": 0.0}
        votes[class_label]["neighbor_count"] += 1
        votes[class_label]["vote_weight"] += float(row["vote_weight"])
    return sorted(votes.values(), key=lambda row: row["vote_weight"], reverse=True)


def _evaluation_rows(y_test: np.ndarray, predictions: np.ndarray) -> TableRows:
    return [
        {"metric": "accuracy", "value": float(accuracy_score(y_test, predictions))},
        {
            "metric": "precision_weighted",
            "value": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        },
        {
            "metric": "recall_weighted",
            "value": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        },
        {
            "metric": "f1_weighted",
            "value": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        },
    ]


def _confusion_rows(y_test: np.ndarray, predictions: np.ndarray) -> TableRows:
    labels = np.unique(np.concatenate([y_test, predictions]))
    matrix = confusion_matrix(y_test, predictions, labels=labels).astype(int)
    return [
        {
            "true_class": _to_python_value(true_label),
            "predicted_class": _to_python_value(predicted_label),
            "count": int(matrix[true_index, predicted_index]),
        }
        for true_index, true_label in enumerate(labels)
        for predicted_index, predicted_label in enumerate(labels)
    ]


def _parameter_rows(params: Mapping[str, object]) -> TableRows:
    keys = ["n_neighbors", "weights", "metric", "p", "algorithm", "leaf_size"]
    return [{"parameter": key, "value": params.get(key)} for key in keys]


def _feature_rows(feature_names: list[str]) -> TableRows:
    return [{"index": index, "feature": name} for index, name in enumerate(feature_names)]


def _to_python_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value

