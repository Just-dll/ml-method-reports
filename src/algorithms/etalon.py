from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

ClassLabel = Hashable


@dataclass(frozen=True)
class EtalonExplanation:
    sample_index: int
    true_class: ClassLabel | None
    predicted_class: ClassLabel
    nearest_etalon_index: int
    nearest_etalon_class: ClassLabel
    distances: dict[ClassLabel, float]
    is_correct: bool | None


class EtalonClassifier(BaseEstimator, ClassifierMixin):
    """Prototype classifier based on one class centroid per label."""

    _SUPPORTED_METRICS = {"euclidean", "manhattan", "chebyshev", "cosine"}
    _SUPPORTED_NORMALIZATIONS = {"standard", "minmax", "maxabs", "robust", "none"}
    _SUPPORTED_PROTOTYPE_STRATEGIES = {"mean", "median", "nearest"}

    def __init__(
        self,
        metric: str = "euclidean",
        normalization: str = "standard",
        prototype_strategy: str = "mean",
        eps: float = 1e-12,
    ) -> None:
        self.metric = metric
        self.normalization = normalization
        self.prototype_strategy = prototype_strategy
        self.eps = eps

    def fit(self, X: ArrayLike, y: ArrayLike) -> EtalonClassifier:
        self._validate_configuration()
        X_checked, y_checked = check_X_y(X, y, dtype=float, ensure_2d=True)
        classes = np.unique(y_checked)
        if len(classes) < 2:
            raise ValueError("EtalonClassifier requires at least two classes in y.")

        self.classes_ = classes
        self.n_features_in_ = int(X_checked.shape[1])
        self.metric_ = self.metric
        self.normalization_ = self.normalization
        self.prototype_strategy_ = self.prototype_strategy
        self.normalization_params_ = self._fit_normalization(X_checked)
        X_normalized = self._transform(X_checked)

        centers = []
        for class_label in self.classes_:
            class_samples = X_normalized[y_checked == class_label]
            if class_samples.size == 0:
                raise RuntimeError(f"Internal error: class {class_label!r} has no samples.")
            centers.append(self._build_prototype(class_samples))
        self.etalon_centers_ = np.vstack(centers)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        distances = self.get_distances(X)
        nearest_indices = np.argmin(distances, axis=1)
        return self.classes_[nearest_indices]

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        distances = self.get_distances(X)
        probabilities = np.zeros_like(distances, dtype=float)

        for row_index, row in enumerate(distances):
            zero_mask = row <= self.eps
            if np.any(zero_mask):
                probabilities[row_index, zero_mask] = 1.0 / float(np.count_nonzero(zero_mask))
                continue

            inverse_distances = 1.0 / (row + self.eps)
            probabilities[row_index] = inverse_distances / inverse_distances.sum()

        return probabilities

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        y_checked = np.asarray(y)
        predictions = self.predict(X)
        if y_checked.shape[0] != predictions.shape[0]:
            raise ValueError("y must contain the same number of samples as X.")
        return float(np.mean(predictions == y_checked))

    def get_distances(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(
            self,
            attributes=["classes_", "etalon_centers_", "n_features_in_", "normalization_"],
        )
        X_checked = check_array(X, dtype=float, ensure_2d=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has the wrong number of features: "
                f"expected {self.n_features_in_}, got {X_checked.shape[1]}."
            )
        X_normalized = self._transform(X_checked)
        return self._calculate_distances(X_normalized)

    def get_class_centers(self) -> dict[ClassLabel, np.ndarray]:
        check_is_fitted(self, attributes=["classes_", "etalon_centers_"])
        return {
            class_label: np.array(center, copy=True)
            for class_label, center in zip(self.classes_, self.etalon_centers_, strict=True)
        }

    def get_params_summary(self) -> dict[str, object]:
        classes = getattr(self, "classes_", None)
        return {
            "metric": self.metric,
            "normalization": self.normalization,
            "prototype_strategy": self.prototype_strategy,
            "classes": None if classes is None else classes.tolist(),
            "n_features": getattr(self, "n_features_in_", None),
        }

    def decision_report(
        self,
        X: ArrayLike,
        y_true: ArrayLike | int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        rows = []
        for explanation in self.explain_batch(X, y_true=y_true, limit=limit):
            row = {
                "sample_index": explanation.sample_index,
                "true_class": self._to_python_value(explanation.true_class),
                "predicted_class": self._to_python_value(explanation.predicted_class),
                "nearest_etalon_class": self._to_python_value(explanation.nearest_etalon_class),
                "is_correct": explanation.is_correct,
            }
            row.update(
                {
                    f"distance_to_{class_label}": distance
                    for class_label, distance in explanation.distances.items()
                }
            )
            rows.append(row)
        return rows

    def _to_python_value(self, value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        return value

    def explain_prediction(
        self,
        X: ArrayLike,
        y_true: ArrayLike | int | None = None,
        sample_index: int = 0,
    ) -> EtalonExplanation:
        if self._looks_like_legacy_index(y_true, sample_index):
            sample_index = int(y_true)
            y_true = None

        distances = self.get_distances(X)
        if sample_index < 0 or sample_index >= distances.shape[0]:
            raise ValueError(
                f"sample_index must be between 0 and {distances.shape[0] - 1}; got {sample_index}."
            )
        true_class = self._get_true_class(y_true, sample_index, distances.shape[0])

        row = distances[sample_index]
        nearest_index = int(np.argmin(row))
        predicted_class = self.classes_[nearest_index]
        is_correct = None if true_class is None else bool(predicted_class == true_class)
        distance_by_class = {
            class_label: float(distance)
            for class_label, distance in zip(self.classes_, row, strict=True)
        }
        return EtalonExplanation(
            sample_index=sample_index,
            true_class=true_class,
            predicted_class=predicted_class,
            nearest_etalon_index=nearest_index,
            nearest_etalon_class=predicted_class,
            distances=distance_by_class,
            is_correct=is_correct,
        )

    def explain_batch(
        self,
        X: ArrayLike,
        y_true: ArrayLike | int | None = None,
        limit: int | None = None,
    ) -> list[EtalonExplanation]:
        if self._looks_like_legacy_limit(y_true, limit):
            limit = int(y_true)
            y_true = None

        distances = self.get_distances(X)
        if limit is not None and limit < 1:
            raise ValueError("limit must be None or an integer greater than or equal to 1.")

        explanation_count = distances.shape[0] if limit is None else min(limit, distances.shape[0])
        return [
            self.explain_prediction(X, y_true=y_true, sample_index=index)
            for index in range(explanation_count)
        ]

    def _get_true_class(
        self,
        y_true: ArrayLike | None,
        sample_index: int,
        expected_length: int,
    ) -> ClassLabel | None:
        if y_true is None:
            return None

        y_checked = np.asarray(y_true)
        if y_checked.shape[0] != expected_length:
            raise ValueError(
                "y_true must contain the same number of samples as X when provided."
            )
        return y_checked[sample_index]

    def _looks_like_legacy_index(self, y_true: object, sample_index: int) -> bool:
        return sample_index == 0 and isinstance(y_true, int | np.integer)

    def _looks_like_legacy_limit(self, y_true: object, limit: int | None) -> bool:
        return limit is None and isinstance(y_true, int | np.integer)

    def _validate_configuration(self) -> None:
        if self.metric not in self._SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric {self.metric!r}. "
                f"Supported metrics: {sorted(self._SUPPORTED_METRICS)}."
            )
        if self.normalization not in self._SUPPORTED_NORMALIZATIONS:
            raise ValueError(
                f"Unsupported normalization {self.normalization!r}. "
                f"Supported normalizations: {sorted(self._SUPPORTED_NORMALIZATIONS)}."
            )
        if self.prototype_strategy not in self._SUPPORTED_PROTOTYPE_STRATEGIES:
            raise ValueError(
                f"Unsupported prototype_strategy {self.prototype_strategy!r}. "
                f"Supported prototype strategies: {sorted(self._SUPPORTED_PROTOTYPE_STRATEGIES)}."
            )
        if self.eps <= 0:
            raise ValueError("eps must be greater than 0.")

    def _fit_normalization(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if self.normalization == "standard":
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            safe_std = np.where(std < self.eps, 1.0, std)
            return {"mean": mean, "std": safe_std}
        if self.normalization == "minmax":
            minimum = np.min(X, axis=0)
            value_range = np.max(X, axis=0) - minimum
            safe_range = np.where(value_range < self.eps, 1.0, value_range)
            return {"min": minimum, "range": safe_range}
        if self.normalization == "maxabs":
            max_abs = np.max(np.abs(X), axis=0)
            safe_max_abs = np.where(max_abs < self.eps, 1.0, max_abs)
            return {"max_abs": safe_max_abs}
        if self.normalization == "robust":
            median = np.median(X, axis=0)
            q75, q25 = np.percentile(X, [75, 25], axis=0)
            iqr = q75 - q25
            safe_iqr = np.where(iqr < self.eps, 1.0, iqr)
            return {"median": median, "iqr": safe_iqr}
        if self.normalization == "none":
            return {}
        raise RuntimeError(f"Internal error: unsupported normalization {self.normalization!r}.")

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "normalization_params_"):
            raise RuntimeError("Internal error: normalization parameters are missing after fit.")

        normalization = self.normalization_
        if normalization == "standard":
            if "mean" not in self.normalization_params_ or "std" not in self.normalization_params_:
                raise RuntimeError("Internal error: standard normalization parameters are missing.")
            return (X - self.normalization_params_["mean"]) / self.normalization_params_["std"]
        if normalization == "minmax":
            if "min" not in self.normalization_params_ or "range" not in self.normalization_params_:
                raise RuntimeError("Internal error: minmax normalization parameters are missing.")
            return (X - self.normalization_params_["min"]) / self.normalization_params_["range"]
        if normalization == "maxabs":
            if "max_abs" not in self.normalization_params_:
                raise RuntimeError("Internal error: maxabs normalization parameters are missing.")
            return X / self.normalization_params_["max_abs"]
        if normalization == "robust":
            if "median" not in self.normalization_params_ or "iqr" not in self.normalization_params_:
                raise RuntimeError("Internal error: robust normalization parameters are missing.")
            return (X - self.normalization_params_["median"]) / self.normalization_params_["iqr"]
        if normalization == "none":
            return X
        raise RuntimeError(f"Internal error: unsupported normalization {normalization!r}.")

    def _build_prototype(self, class_samples: np.ndarray) -> np.ndarray:
        if self.prototype_strategy == "mean":
            return np.mean(class_samples, axis=0)
        if self.prototype_strategy == "median":
            return np.median(class_samples, axis=0)
        if self.prototype_strategy == "nearest":
            class_mean = np.mean(class_samples, axis=0)
            nearest_index = int(np.argmin(self._calculate_point_distances(class_samples, class_mean)))
            return class_samples[nearest_index]
        raise RuntimeError(
            f"Internal error: unsupported prototype_strategy {self.prototype_strategy!r}."
        )

    def _calculate_distances(self, X: np.ndarray) -> np.ndarray:
        differences = X[:, np.newaxis, :] - self.etalon_centers_[np.newaxis, :, :]
        if self.metric == "euclidean":
            return np.linalg.norm(differences, axis=2)
        if self.metric == "manhattan":
            return np.sum(np.abs(differences), axis=2)
        if self.metric == "chebyshev":
            return np.max(np.abs(differences), axis=2)
        if self.metric == "cosine":
            numerator = X @ self.etalon_centers_.T
            x_norm = np.linalg.norm(X, axis=1)
            center_norm = np.linalg.norm(self.etalon_centers_, axis=1)
            denominator = x_norm[:, np.newaxis] * center_norm[np.newaxis, :]
            safe_denominator = np.where(denominator < self.eps, 1.0, denominator)
            return 1.0 - (numerator / safe_denominator)
        raise RuntimeError(f"Internal error: unsupported metric {self.metric!r}.")

    def _calculate_point_distances(self, X: np.ndarray, point: np.ndarray) -> np.ndarray:
        differences = X - point
        if self.metric == "euclidean":
            return np.linalg.norm(differences, axis=1)
        if self.metric == "manhattan":
            return np.sum(np.abs(differences), axis=1)
        if self.metric == "chebyshev":
            return np.max(np.abs(differences), axis=1)
        if self.metric == "cosine":
            numerator = X @ point
            denominator = np.linalg.norm(X, axis=1) * np.linalg.norm(point)
            safe_denominator = np.where(denominator < self.eps, 1.0, denominator)
            return 1.0 - (numerator / safe_denominator)
        raise RuntimeError(f"Internal error: unsupported metric {self.metric!r}.")

