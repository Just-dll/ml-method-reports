from __future__ import annotations

import numpy as np
import pandas as pd
from ml_method_reports.reporting.context import ReportContext
from ml_method_reports.reporting.models import ExperimentReport, ReportSection
from ml_method_reports.reporting.serialization import sanitize_table_rows, to_report_value
from ml_method_reports.reporting.types import (
    FeatureMatrix,
    ParametrizedModel,
    PredictiveModel,
    ReportValue,
    TableRows,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class GenericClassificationReportBuilder:
    def __init__(self, context: ReportContext) -> None:
        self._context = context

    def build(self) -> ExperimentReport:
        context = self._context
        predictions = self._predictions()
        names = self._feature_names()
        metadata = {
            "model": context.model_name,
            "model_class": type(context.model).__name__,
            "dataset_source": context.dataset_source,
            "target": context.target_name,
            "features": len(names),
            **dict(context.metadata),
        }

        sections = [
            ReportSection(title="1. Model Overview", table=self._overview_rows(names)),
            ReportSection(title="2. Model Parameters", table=self._parameter_rows()),
            ReportSection(title="3. Data Summary", table=self._data_summary_rows(names)),
            ReportSection(title="4. Prediction Samples", table=self._prediction_sample_rows(predictions)),
        ]

        if context.y_test is not None:
            sections.extend(
                [
                    ReportSection(title="5. Evaluation Metrics", table=self._evaluation_rows(predictions)),
                    ReportSection(title="6. Confusion Matrix", table=self._confusion_rows(predictions)),
                    ReportSection(
                        title="7. Classification Report",
                        table=self._classification_report_rows(predictions),
                    ),
                ]
            )

        feature_importance_rows = self._feature_importance_rows(names)
        if feature_importance_rows:
            sections.append(
                ReportSection(title="8. Feature Importances", table=feature_importance_rows)
            )

        coefficient_rows = self._coefficient_rows(names)
        if coefficient_rows:
            sections.append(ReportSection(title="9. Coefficients", table=coefficient_rows))

        sections.append(
            ReportSection(
                title="Method-specific Explanation",
                content=(
                    "A method-specific educational explanation is unavailable for this model. "
                    "This generic report summarizes model outputs and standard classification metrics."
                ),
            )
        )

        return ExperimentReport(
            title=f"{context.model_name} Generic Classification Report",
            subtitle="Model-agnostic report generated from a sklearn-like classifier interface.",
            metadata=metadata,
            sections=sections,
        )

    def _predictions(self) -> np.ndarray:
        if self._context.predictions is not None:
            return np.asarray(self._context.predictions)
        if not isinstance(self._context.model, PredictiveModel):
            raise ValueError("Generic classification reports require a model with a predict(X) method.")
        return np.asarray(self._context.model.predict(self._context.X_test))

    def _feature_names(self) -> list[str]:
        if self._context.feature_names is not None:
            return list(self._context.feature_names)
        if isinstance(self._context.X_test, pd.DataFrame):
            return self._context.X_test.columns.astype(str).tolist()
        shape = np.asarray(self._context.X_test).shape
        feature_count = int(shape[1]) if len(shape) > 1 else 1
        return [f"feature_{index}" for index in range(feature_count)]

    def _overview_rows(self, names: list[str]) -> TableRows:
        return [
            {"item": "model name", "value": self._context.model_name},
            {"item": "model class", "value": type(self._context.model).__name__},
            {"item": "dataset source", "value": self._context.dataset_source},
            {"item": "target", "value": self._context.target_name},
            {"item": "feature count", "value": len(names)},
        ]

    def _parameter_rows(self) -> TableRows:
        if not isinstance(self._context.model, ParametrizedModel):
            return [{"parameter": "get_params", "value": "not available"}]
        params = self._context.model.get_params()
        return [
            {"parameter": key, "value": to_report_value(value)}
            for key, value in sorted(params.items(), key=lambda item: str(item[0]))
        ]

    def _data_summary_rows(self, names: list[str]) -> TableRows:
        rows = [
            {"item": "train samples", "value": self._sample_count(self._context.X_train)},
            {"item": "test samples", "value": self._sample_count(self._context.X_test)},
            {"item": "features", "value": names},
        ]
        if self._context.y_train is not None:
            rows.append({"item": "train target samples", "value": len(np.asarray(self._context.y_train))})
        if self._context.y_test is not None:
            rows.append({"item": "test target samples", "value": len(np.asarray(self._context.y_test))})
        return rows

    def _prediction_sample_rows(self, predictions: np.ndarray, limit: int = 10) -> TableRows:
        rows = []
        y_true = None if self._context.y_test is None else np.asarray(self._context.y_test)
        for index, predicted in enumerate(predictions[:limit]):
            row = {"sample_index": index, "predicted": self._to_python(predicted)}
            if y_true is not None:
                row["true"] = self._to_python(y_true[index])
                row["is_correct"] = bool(predicted == y_true[index])
            rows.append(row)
        return rows

    def _evaluation_rows(self, predictions: np.ndarray) -> TableRows:
        y_true = self._checked_y_true(predictions)
        return [
            {"metric": "accuracy", "value": float(accuracy_score(y_true, predictions))},
            {
                "metric": "precision_weighted",
                "value": float(precision_score(y_true, predictions, average="weighted", zero_division=0)),
            },
            {
                "metric": "recall_weighted",
                "value": float(recall_score(y_true, predictions, average="weighted", zero_division=0)),
            },
            {
                "metric": "f1_weighted",
                "value": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
            },
        ]

    def _confusion_rows(self, predictions: np.ndarray) -> TableRows:
        y_true = self._checked_y_true(predictions)
        labels = np.unique(np.concatenate([y_true, predictions]))
        matrix = confusion_matrix(y_true, predictions, labels=labels).astype(int)
        return [
            {
                "true_class": self._to_python(true_label),
                "predicted_class": self._to_python(predicted_label),
                "count": int(matrix[true_index, predicted_index]),
            }
            for true_index, true_label in enumerate(labels)
            for predicted_index, predicted_label in enumerate(labels)
        ]

    def _classification_report_rows(self, predictions: np.ndarray) -> TableRows:
        y_true = self._checked_y_true(predictions)
        report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
        return sanitize_table_rows(
            [
                {"class": label, **values} if isinstance(values, dict) else {"class": label, "value": values}
                for label, values in report.items()
            ]
        )

    def _feature_importance_rows(self, names: list[str]) -> TableRows:
        values = getattr(self._context.model, "feature_importances_", None)
        if values is None:
            return []
        scores = np.asarray(values, dtype=float).ravel()
        if len(scores) != len(names):
            return []
        return sorted(
            [
                {"feature": name, "importance": float(score)}
                for name, score in zip(names, scores, strict=True)
            ],
            key=lambda row: abs(row["importance"]),
            reverse=True,
        )

    def _coefficient_rows(self, names: list[str]) -> TableRows:
        values = getattr(self._context.model, "coef_", None)
        if values is None:
            return []
        coefficients = np.asarray(values, dtype=float)
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)
        if coefficients.shape[1] != len(names):
            return []
        classes = getattr(self._context.model, "classes_", range(coefficients.shape[0]))
        rows = []
        for class_index, class_coefficients in enumerate(coefficients):
            class_label = classes[class_index] if len(classes) == coefficients.shape[0] else class_index
            for name, coefficient in zip(names, class_coefficients, strict=True):
                rows.append(
                    {
                        "class": self._to_python(class_label),
                        "feature": name,
                        "coefficient": float(coefficient),
                    }
                )
        return sorted(rows, key=lambda row: abs(row["coefficient"]), reverse=True)

    def _checked_y_true(self, predictions: np.ndarray) -> np.ndarray:
        if self._context.y_test is None:
            raise ValueError("y_test is required for evaluation metrics.")
        y_true = np.asarray(self._context.y_test)
        if y_true.shape[0] != predictions.shape[0]:
            raise ValueError(
                "y_test and predictions must contain the same number of samples: "
                f"got {y_true.shape[0]} and {predictions.shape[0]}."
            )
        return y_true

    def _sample_count(self, value: FeatureMatrix | None) -> int | None:
        if value is None:
            return None
        return int(np.asarray(value).shape[0])

    def _to_python(self, value: object) -> ReportValue:
        if isinstance(value, np.generic):
            return value.item()
        return value
