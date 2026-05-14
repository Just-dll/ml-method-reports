from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml_method_reports.reporting.models import ReportSection
from ml_method_reports.reporting.plots import plot_bar_chart
from ml_method_reports.reporting.types import (
    FeatureMatrix,
    ParametrizedModel,
    PredictionVector,
    PredictiveModel,
    ScalingParams,
    TableRows,
    TargetVector,
)


def as_2d_array(value: FeatureMatrix) -> np.ndarray:
    return np.asarray(value, dtype=float)


def as_1d_array(value: TargetVector | PredictionVector) -> np.ndarray:
    return np.asarray(value)


def feature_names(X: FeatureMatrix, provided: list[str] | None = None) -> list[str]:
    if provided is not None:
        return list(provided)
    if isinstance(X, pd.DataFrame):
        return X.columns.astype(str).tolist()
    feature_count = as_2d_array(X).shape[1]
    return [f"feature_{index}" for index in range(feature_count)]


def selected_index(index: int, sample_count: int) -> int:
    if index < 0 or index >= sample_count:
        raise ValueError(f"selected_sample_index must be between 0 and {sample_count - 1}; got {index}.")
    return int(index)


def selected_query(original_X: FeatureMatrix, X_array: np.ndarray, index: int) -> FeatureMatrix:
    if isinstance(original_X, pd.DataFrame):
        return original_X.iloc[[index]]
    return X_array[[index]]


def predictions(model: PredictiveModel, X_query: FeatureMatrix, provided: PredictionVector | None = None) -> np.ndarray:
    if provided is not None:
        return np.asarray(provided)
    return np.asarray(model.predict(X_query))


def model_params_rows(model: object, keys: list[str]) -> TableRows:
    params = model.get_params() if isinstance(model, ParametrizedModel) else {}
    return [{"parameter": key, "value": params.get(key)} for key in keys]


def preprocessing_section(method: str = "none", params: ScalingParams | None = None, names: list[str] | None = None) -> ReportSection:
    if method == "standard":
        content = (
            "This report uses standard scaling before model fitting. Scaling affects "
            "distance-based models directly and makes coefficient magnitudes easier to compare."
        )
        table = scaling_rows(params, names or [])
    elif method == "none":
        content = "No explicit preprocessing metadata was provided for this report."
        table = [{"preprocessing": "none", "details": "exact preprocessing parameters not provided"}]
    else:
        content = f"This report received preprocessing metadata labeled {method!r}."
        table = [{"preprocessing": method, "details": "see upstream pipeline metadata"}]
    return ReportSection(title="2. Preprocessing / Scaling Summary", content=content, table=table)


def scaling_rows(params: ScalingParams | None, names: list[str]) -> TableRows:
    if not params:
        return [{"scaling": "standard", "details": "StandardScaler parameters not provided"}]
    means = params.get("mean")
    scales = params.get("scale")
    if scales is None:
        scales = params.get("std")
    if means is None or scales is None:
        return [{"scaling": "standard", "details": "mean/std not provided"}]
    return [
        {"feature": name, "mean": float(mean), "std": float(scale)}
        for name, mean, scale in zip(names, means, scales, strict=True)
    ]


def evaluation_rows(y_true: TargetVector, y_pred: PredictionVector) -> TableRows:
    true_values = as_1d_array(y_true)
    predicted_values = as_1d_array(y_pred)
    return [
        {"metric": "accuracy", "value": float(accuracy_score(true_values, predicted_values))},
        {"metric": "precision_weighted", "value": float(precision_score(true_values, predicted_values, average="weighted", zero_division=0))},
        {"metric": "recall_weighted", "value": float(recall_score(true_values, predicted_values, average="weighted", zero_division=0))},
        {"metric": "f1_weighted", "value": float(f1_score(true_values, predicted_values, average="weighted", zero_division=0))},
    ]


def confusion_rows(y_true: TargetVector, y_pred: PredictionVector) -> TableRows:
    true_values = as_1d_array(y_true)
    predicted_values = as_1d_array(y_pred)
    labels = np.unique(np.concatenate([true_values, predicted_values]))
    matrix = confusion_matrix(true_values, predicted_values, labels=labels).astype(int)
    return [
        {
            "true_class": to_python(true_label),
            "predicted_class": to_python(predicted_label),
            "count": int(matrix[true_index, predicted_index]),
        }
        for true_index, true_label in enumerate(labels)
        for predicted_index, predicted_label in enumerate(labels)
    ]


def total_errors(y_true: TargetVector, y_pred: PredictionVector) -> int:
    return int(np.count_nonzero(as_1d_array(y_true) != as_1d_array(y_pred)))


def error_analysis_text(rows: TableRows) -> str:
    mistakes = [row for row in rows if row["true_class"] != row["predicted_class"] and row["count"] > 0]
    error_count = sum(int(row["count"]) for row in mistakes)
    if error_count == 0:
        return "There are 0 recognition errors on the evaluation split."
    detail = "; ".join(
        f"class {row['true_class']} predicted as {row['predicted_class']}: {row['count']}"
        for row in mistakes
    )
    return f"There are {error_count} recognition error(s) on the evaluation split. {detail}."


def probability_rows(model: object, query: FeatureMatrix) -> TableRows:
    if not hasattr(model, "predict_proba"):
        return []
    probabilities = model.predict_proba(query)[0]
    classes = getattr(model, "classes_", range(len(probabilities)))
    return [
        {"class": to_python(class_label), "probability": float(probability)}
        for class_label, probability in zip(classes, probabilities, strict=True)
    ]


def score_rows(model: object, query: FeatureMatrix) -> TableRows:
    if not hasattr(model, "decision_function"):
        return []
    scores = np.asarray(model.decision_function(query)).ravel()
    if scores.shape[0] == 1:
        return [{"score": "decision_function", "value": float(scores[0])}]
    classes = getattr(model, "classes_", range(len(scores)))
    return [
        {"class": to_python(class_label), "decision_score": float(score)}
        for class_label, score in zip(classes, scores, strict=True)
    ]


def feature_score_rows(names: list[str], scores: ArrayLike, *, key: str = "score", absolute: bool = False) -> TableRows:
    values = np.asarray(scores, dtype=float).ravel()
    if values.shape[0] != len(names):
        return []
    rows = [
        {"feature": name, key: float(abs(value) if absolute else value)}
        for name, value in zip(names, values, strict=True)
    ]
    return sorted(rows, key=lambda row: abs(float(row[key])), reverse=True)


def build_bar_asset(rows: TableRows, assets_dir: Path, filename: str, *, label_key: str, value_key: str, title: str) -> dict[str, str]:
    if not rows:
        return {}
    path = plot_bar_chart(rows[:12], label_key=label_key, value_key=value_key, output_path=assets_dir / filename, title=title)
    return {Path(filename).stem: str(Path("assets") / path.name)}


def table_or_note(title: str, rows: TableRows, note: str) -> ReportSection:
    if rows:
        return ReportSection(title=title, table=rows)
    return ReportSection(title=title, content=note)


def class_distribution_rows(labels: TargetVector, label_name: str = "class") -> TableRows:
    values, counts = np.unique(as_1d_array(labels), return_counts=True)
    return [
        {label_name: to_python(value), "count": int(count)}
        for value, count in zip(values, counts, strict=True)
    ]


def to_python(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value

