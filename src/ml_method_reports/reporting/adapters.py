from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_method_reports.algorithms import EtalonClassifier, evaluate_predictions
from ml_method_reports.reporting.builders import (
    AgglomerativeReportBuilder,
    AgglomerativeReportInput,
    DecisionTreeReportBuilder,
    DecisionTreeReportInput,
    EtalonReportBuilder,
    EtalonReportInput,
    EtalonRun,
    KMeansReportBuilder,
    KMeansReportInput,
    KnnReportBuilder,
    KnnReportInput,
    LogisticRegressionReportBuilder,
    LogisticRegressionReportInput,
    RandomForestReportBuilder,
    RandomForestReportInput,
    SvcReportBuilder,
    SvcReportInput,
)
from ml_method_reports.reporting.context import ReportContext
from ml_method_reports.reporting.models import ExperimentReport
from ml_method_reports.reporting.types import TableRows

ReportType = Literal[
    "auto",
    "generic",
    "etalon",
    "knn",
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "svc",
    "kmeans",
    "agglomerative",
]


@dataclass(frozen=True, slots=True)
class ReportAdapter:
    report_type: str
    model_type: type
    default_stem: str
    requires_test_data: bool = True
    requires_supervised_split: bool = False

    def supports(self, context: ReportContext) -> bool:
        return isinstance(context.model, self.model_type)

    def has_required_data(self, context: ReportContext) -> bool:
        if self.requires_test_data and context.X_test is None:
            return False
        if not self.requires_supervised_split:
            return True
        return (
            context.X_train is not None
            and context.y_train is not None
            and context.y_test is not None
        )

    def build(self, context: ReportContext) -> ExperimentReport:
        raise NotImplementedError


class EtalonReportAdapter(ReportAdapter):
    def __init__(self) -> None:
        super().__init__(
            "etalon",
            EtalonClassifier,
            "etalon_report",
            True,
            requires_supervised_split=True,
        )

    def build(self, context: ReportContext) -> ExperimentReport:
        _require_supervised_split(context, self.report_type)
        model = context.model
        if not isinstance(model, EtalonClassifier):
            raise TypeError("Etalon report requires EtalonClassifier.")

        X_train = _as_frame(context.X_train, context.feature_names)
        X_test = _as_frame(context.X_test, context.feature_names)
        y_train = np.asarray(context.y_train)
        y_test = np.asarray(context.y_test)
        feature_names = list(X_train.columns)
        feature_index = int(context.options.get("feature_index", 0))
        if feature_index < 0 or feature_index >= len(feature_names):
            raise ValueError(
                f"feature_index must be between 0 and {len(feature_names) - 1}; got {feature_index}."
            )

        predictions = _predictions(context, X_test)
        full_run = EtalonRun(
            mode="Full-feature experiment",
            model=model,
            X_train=X_train.to_numpy(),
            X_test=X_test.to_numpy(),
            y_train=y_train,
            y_test=y_test,
            predictions=predictions,
            feature_names=feature_names,
            summary=evaluate_predictions(y_test, predictions),
            test_indices=np.arange(len(X_test)),
        )

        one_feature_name = feature_names[feature_index]
        one_feature_model = EtalonClassifier(
            metric=model.metric,
            normalization=model.normalization,
            prototype_strategy=model.prototype_strategy,
            eps=model.eps,
        ).fit(X_train.iloc[:, [feature_index]].to_numpy(), y_train)
        one_feature_predictions = one_feature_model.predict(X_test.iloc[:, [feature_index]].to_numpy())
        one_feature_run = EtalonRun(
            mode="One-feature experiment",
            model=one_feature_model,
            X_train=X_train.iloc[:, [feature_index]].to_numpy(),
            X_test=X_test.iloc[:, [feature_index]].to_numpy(),
            y_train=y_train,
            y_test=y_test,
            predictions=one_feature_predictions,
            feature_names=[one_feature_name],
            summary=evaluate_predictions(y_test, one_feature_predictions),
            test_indices=np.arange(len(X_test)),
        )

        dataframe = pd.concat(
            [
                X_train.assign(**{context.target_name: y_train}),
                X_test.assign(**{context.target_name: y_test}),
            ],
            ignore_index=True,
        )
        train_indices = np.arange(len(X_train))
        test_indices = np.arange(len(X_train), len(X_train) + len(X_test))
        comparison_rows = _option_table_rows(context, "comparison_rows")
        if not comparison_rows:
            comparison_rows = [
                {
                    "model": type(model).__name__,
                    "type": "custom",
                    "accuracy": full_run.summary.accuracy,
                    "error_rate": full_run.summary.error_rate,
                }
            ]

        report_input = EtalonReportInput(
            dataframe=dataframe,
            train_indices=train_indices,
            test_indices=test_indices,
            full_run=full_run,
            one_feature_run=one_feature_run,
            comparison_rows=comparison_rows,
            metric=model.metric,
            normalization=model.normalization,
            prototype_strategy=model.prototype_strategy,
            feature_index=feature_index,
            one_feature_name=one_feature_name,
            random_state=int(context.options.get("random_state", 0)),
            dataset_source=context.dataset_source,
            target_name=context.target_name,
        )
        return EtalonReportBuilder(report_input, assets_dir=context.assets_dir).build()


class KnnReportAdapter(ReportAdapter):
    def __init__(self) -> None:
        super().__init__("knn", KNeighborsClassifier, "knn_report", requires_supervised_split=True)

    def build(self, context: ReportContext) -> ExperimentReport:
        _require_supervised_split(context, self.report_type)
        report_input = KnnReportInput(
            model=context.model,
            X_train=context.X_train,
            X_test=context.X_test,
            y_train=context.y_train,
            y_test=context.y_test,
            feature_names=context.feature_names,
            predictions=context.predictions,
            leaderboard_rows=_option_table_rows(context, "leaderboard_rows"),
            dataset_source=context.dataset_source,
            target_name=context.target_name,
            selected_sample_index=_selected_sample_index(context),
            selected_sample_indices=_option_int_list(context, "selected_sample_indices"),
            scaling_method=_option_str(context, "scaling_method", "none"),
            scaling_params=context.options.get("scaling_params"),
        )
        return KnnReportBuilder(report_input, assets_dir=context.assets_dir).build()


class LogisticRegressionReportAdapter(ReportAdapter):
    def __init__(self) -> None:
        super().__init__(
            "logistic_regression",
            LogisticRegression,
            "logistic_regression_report",
            True,
            requires_supervised_split=True,
        )

    def build(self, context: ReportContext) -> ExperimentReport:
        _require_supervised_split(context, self.report_type)
        report_input = LogisticRegressionReportInput(
            model=context.model,
            X_train=context.X_train,
            X_test=context.X_test,
            y_train=context.y_train,
            y_test=context.y_test,
            feature_names=context.feature_names,
            predictions=context.predictions,
            dataset_source=context.dataset_source,
            target_name=context.target_name,
            selected_sample_index=_selected_sample_index(context),
            scaling_method=_option_str(context, "scaling_method", "none"),
            scaling_params=context.options.get("scaling_params"),
        )
        return LogisticRegressionReportBuilder(report_input, assets_dir=context.assets_dir).build()


class DecisionTreeReportAdapter(ReportAdapter):
    def __init__(self) -> None:
        super().__init__(
            "decision_tree",
            DecisionTreeClassifier,
            "decision_tree_report",
            True,
            requires_supervised_split=True,
        )

    def build(self, context: ReportContext) -> ExperimentReport:
        _require_supervised_split(context, self.report_type)
        report_input = DecisionTreeReportInput(
            model=context.model,
            X_train=context.X_train,
            X_test=context.X_test,
            y_train=context.y_train,
            y_test=context.y_test,
            feature_names=context.feature_names,
            predictions=context.predictions,
            dataset_source=context.dataset_source,
            target_name=context.target_name,
            selected_sample_index=_selected_sample_index(context),
            preprocessing_method=_option_str(context, "preprocessing_method", "none"),
            preprocessing_params=context.options.get("preprocessing_params"),
        )
        return DecisionTreeReportBuilder(report_input, assets_dir=context.assets_dir).build()


class RandomForestReportAdapter(ReportAdapter):
    def __init__(self) -> None:
        super().__init__(
            "random_forest",
            RandomForestClassifier,
            "random_forest_report",
            True,
            requires_supervised_split=True,
        )

    def build(self, context: ReportContext) -> ExperimentReport:
        _require_supervised_split(context, self.report_type)
        report_input = RandomForestReportInput(
            model=context.model,
            X_train=context.X_train,
            X_test=context.X_test,
            y_train=context.y_train,
            y_test=context.y_test,
            feature_names=context.feature_names,
            predictions=context.predictions,
            dataset_source=context.dataset_source,
            target_name=context.target_name,
            selected_sample_index=_selected_sample_index(context),
            preprocessing_method=_option_str(context, "preprocessing_method", "none"),
            preprocessing_params=context.options.get("preprocessing_params"),
        )
        return RandomForestReportBuilder(report_input, assets_dir=context.assets_dir).build()


class SvcReportAdapter(ReportAdapter):
    def __init__(self) -> None:
        super().__init__("svc", SVC, "svc_report", requires_supervised_split=True)

    def build(self, context: ReportContext) -> ExperimentReport:
        _require_supervised_split(context, self.report_type)
        report_input = SvcReportInput(
            model=context.model,
            X_train=context.X_train,
            X_test=context.X_test,
            y_train=context.y_train,
            y_test=context.y_test,
            feature_names=context.feature_names,
            predictions=context.predictions,
            dataset_source=context.dataset_source,
            target_name=context.target_name,
            selected_sample_index=_selected_sample_index(context),
            scaling_method=_option_str(context, "scaling_method", "none"),
            scaling_params=context.options.get("scaling_params"),
        )
        return SvcReportBuilder(report_input, assets_dir=context.assets_dir).build()


class KMeansReportAdapter(ReportAdapter):
    def __init__(self) -> None:
        super().__init__("kmeans", KMeans, "kmeans_report")

    def build(self, context: ReportContext) -> ExperimentReport:
        report_input = KMeansReportInput(
            model=context.model,
            X=context.X_test,
            feature_names=context.feature_names,
            true_labels=context.y_test,
            dataset_source=context.dataset_source,
            selected_sample_index=_selected_sample_index(context),
            scaling_method=_option_str(context, "scaling_method", "none"),
            scaling_params=context.options.get("scaling_params"),
        )
        return KMeansReportBuilder(report_input, assets_dir=context.assets_dir).build()


class AgglomerativeReportAdapter(ReportAdapter):
    def __init__(self) -> None:
        super().__init__("agglomerative", AgglomerativeClustering, "agglomerative_report")

    def build(self, context: ReportContext) -> ExperimentReport:
        report_input = AgglomerativeReportInput(
            model=context.model,
            X=context.X_test,
            feature_names=context.feature_names,
            true_labels=context.y_test,
            dataset_source=context.dataset_source,
            scaling_method=_option_str(context, "scaling_method", "none"),
            scaling_params=context.options.get("scaling_params"),
        )
        return AgglomerativeReportBuilder(report_input, assets_dir=context.assets_dir).build()


ADAPTERS: tuple[ReportAdapter, ...] = (
    EtalonReportAdapter(),
    KnnReportAdapter(),
    LogisticRegressionReportAdapter(),
    DecisionTreeReportAdapter(),
    RandomForestReportAdapter(),
    SvcReportAdapter(),
    KMeansReportAdapter(),
    AgglomerativeReportAdapter(),
)


def resolve_report_adapter(
    context: ReportContext,
    report_type: ReportType,
) -> ReportAdapter | None:
    if report_type == "generic":
        return None
    if report_type == "auto":
        return next(
            (
                adapter
                for adapter in ADAPTERS
                if adapter.supports(context) and adapter.has_required_data(context)
            ),
            None,
        )

    for adapter in ADAPTERS:
        if adapter.report_type == report_type:
            if not adapter.supports(context):
                raise ValueError(
                    f"report_type={report_type!r} does not support model "
                    f"{type(context.model).__name__}."
                )
            if not adapter.has_required_data(context):
                return None
            return adapter
    raise ValueError(f"Unsupported report_type {report_type!r}.")


def _require_supervised_split(context: ReportContext, report_type: str) -> None:
    if (
        context.X_train is None
        or context.X_test is None
        or context.y_train is None
        or context.y_test is None
    ):
        raise ValueError(
            f"report_type={report_type!r} requires X_train, X_test, y_train, and y_test."
        )


def _predictions(context: ReportContext, X_test: pd.DataFrame) -> np.ndarray:
    if context.predictions is not None:
        return np.asarray(context.predictions)
    return np.asarray(context.model.predict(X_test))


def _as_frame(value: object, feature_names: list[str] | None) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError("Feature data must be two-dimensional.")
    names = feature_names or [f"feature_{index}" for index in range(array.shape[1])]
    return pd.DataFrame(array, columns=names)


def _selected_sample_index(context: ReportContext) -> int:
    return int(context.options.get("selected_sample_index", 0))


def _option_str(context: ReportContext, key: str, default: str) -> str:
    value = context.options.get(key, default)
    return str(value)


def _option_table_rows(context: ReportContext, key: str) -> TableRows | None:
    value = context.options.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"options[{key!r}] must be a list of table rows.")
    return value


def _option_int_list(context: ReportContext, key: str) -> list[int] | None:
    value = context.options.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"options[{key!r}] must be a list of integers.")
    return [int(item) for item in value]
