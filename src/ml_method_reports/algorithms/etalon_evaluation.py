from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import confusion_matrix

ClassLabel = Hashable


@dataclass(frozen=True, slots=True)
class EtalonEvaluationSummary:
    total_samples: int
    correct_count: int
    error_count: int
    accuracy: float
    error_rate: float
    probability_correct: float
    probability_error: float
    confusion_matrix: list[list[int]]
    classes: list[ClassLabel]


def evaluate_predictions(y_true: ArrayLike, y_pred: ArrayLike) -> EtalonEvaluationSummary:
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    if true_values.shape[0] != predicted_values.shape[0]:
        raise ValueError("y_true and y_pred must contain the same number of samples.")

    total = int(true_values.shape[0])
    if total < 1:
        raise ValueError("y_true and y_pred must contain at least one sample.")

    correct = int(np.count_nonzero(true_values == predicted_values))
    errors = total - correct
    accuracy = correct / total
    error_rate = errors / total
    classes = np.unique(np.concatenate([true_values, predicted_values])).tolist()

    return EtalonEvaluationSummary(
        total_samples=total,
        correct_count=correct,
        error_count=errors,
        accuracy=accuracy,
        error_rate=error_rate,
        probability_correct=accuracy,
        probability_error=error_rate,
        confusion_matrix=confusion_matrix(
            true_values,
            predicted_values,
            labels=classes,
        ).astype(int).tolist(),
        classes=classes,
    )

