from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from ml_method_reports.reporting.adapters import ReportAdapter, ReportType, resolve_report_adapter
from ml_method_reports.reporting.builders.generic import GenericClassificationReportBuilder
from ml_method_reports.reporting.context import ReportContext
from ml_method_reports.reporting.models import ExperimentReport
from ml_method_reports.reporting.saving import save_html_report, save_pdf_report, save_report_bundle
from ml_method_reports.reporting.types import (
    FeatureMatrix,
    PathLike,
    PredictionVector,
    ReportBuilderObject,
    TargetVector,
)

ReportFactory = Callable[[ReportContext], ExperimentReport]
ReportBuilderInput = (
    type[ReportBuilderObject]
    | ReportBuilderObject
    | ExperimentReport
    | Callable[[ReportContext], ExperimentReport | ReportBuilderObject]
)


def report_for(model: object, *, model_name: str | None = None) -> ReportRequest:
    return ReportRequest(model=model, model_name=model_name)


class ReportRequest:
    def __init__(self, model: object, model_name: str | None = None) -> None:
        self._model = model
        self._model_name = model_name
        self._X_train: FeatureMatrix | None = None
        self._X_test: FeatureMatrix | None = None
        self._y_train: TargetVector | None = None
        self._y_test: TargetVector | None = None
        self._predictions: PredictionVector | None = None
        self._feature_names: list[str] | None = None
        self._target_name = "target"
        self._metadata: dict[str, object] = {}
        self._options: dict[str, object] = {}
        self._report_type: ReportType = "auto"
        self._builder: ReportBuilderInput | None = None
        self._report_factory: ReportFactory | None = None

    def with_data(
        self,
        *,
        X_test: FeatureMatrix | None = None,
        X_train: FeatureMatrix | None = None,
        y_train: TargetVector | None = None,
        y_test: TargetVector | None = None,
        predictions: PredictionVector | None = None,
        feature_names: list[str] | None = None,
        target_name: str = "target",
    ) -> ReportRequest:
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._predictions = predictions
        self._feature_names = feature_names
        self._target_name = target_name
        return self

    def with_training_data(
        self,
        *,
        X_train: FeatureMatrix,
        y_train: TargetVector | None = None,
        feature_names: list[str] | None = None,
        target_name: str | None = None,
    ) -> ReportRequest:
        self._X_train = X_train
        self._y_train = y_train
        if feature_names is not None:
            self._feature_names = feature_names
        if target_name is not None:
            self._target_name = target_name
        return self

    def with_train_data(
        self,
        *,
        X_train: FeatureMatrix,
        y_train: TargetVector | None = None,
        feature_names: list[str] | None = None,
        target_name: str | None = None,
    ) -> ReportRequest:
        return self.with_training_data(
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            target_name=target_name,
        )

    def with_test_data(
        self,
        *,
        X_test: FeatureMatrix,
        y_test: TargetVector | None = None,
        predictions: PredictionVector | None = None,
        feature_names: list[str] | None = None,
        target_name: str | None = None,
    ) -> ReportRequest:
        self._X_test = X_test
        self._y_test = y_test
        self._predictions = predictions
        if feature_names is not None:
            self._feature_names = feature_names
        if target_name is not None:
            self._target_name = target_name
        return self

    def with_evaluation_data(
        self,
        *,
        X_test: FeatureMatrix,
        y_test: TargetVector | None = None,
        predictions: PredictionVector | None = None,
        feature_names: list[str] | None = None,
        target_name: str | None = None,
    ) -> ReportRequest:
        return self.with_test_data(
            X_test=X_test,
            y_test=y_test,
            predictions=predictions,
            feature_names=feature_names,
            target_name=target_name,
        )

    def with_options(self, **options: object) -> ReportRequest:
        self._options.update(options)
        return self

    def with_metadata(self, **metadata: object) -> ReportRequest:
        self._metadata.update(metadata)
        return self

    def as_auto(self) -> ReportRequest:
        self._report_type = "auto"
        return self

    def as_generic(self) -> ReportRequest:
        self._report_type = "generic"
        return self

    def as_type(self, report_type: ReportType) -> ReportRequest:
        self._report_type = report_type
        return self

    def with_builder(self, builder: ReportBuilderInput) -> ReportRequest:
        if self._report_factory is not None:
            raise ValueError("Pass either builder or report_factory, not both.")
        self._builder = builder
        return self

    def with_report_factory(self, report_factory: ReportFactory) -> ReportRequest:
        if self._builder is not None:
            raise ValueError("Pass either builder or report_factory, not both.")
        self._report_factory = report_factory
        return self

    def build(self) -> ExperimentReport:
        context = self._context(output_dir=Path("."), assets_dir=None)
        report, _ = _build_report(
            context,
            builder=self._builder,
            report_factory=self._report_factory,
            report_type=self._report_type,
        )
        return report

    def to_html(self, *, embed_images: bool = False) -> str:
        from ml_method_reports.reporting.html_report import HtmlReportGenerator

        if embed_images:
            with TemporaryDirectory(prefix="ml-method-reports-") as temporary_dir:
                output = Path(temporary_dir)
                report = self._build_with_context(output_dir=output, assets_dir=output / "assets")[0]
                return HtmlReportGenerator().render(report, embed_images=True, base_dir=output)

        report = self.build()
        return HtmlReportGenerator().render(report, embed_images=embed_images)

    def display(self) -> None:
        from ml_method_reports.reporting.notebook import display_report

        with TemporaryDirectory(prefix="ml-method-reports-") as temporary_dir:
            output = Path(temporary_dir)
            report, _ = self._build_with_context(output_dir=output, assets_dir=output / "assets")
            display_report(report, base_dir=output)

    def save(self, output_dir: PathLike, *, stem: str | None = None) -> tuple[Path, Path]:
        output = Path(output_dir)
        report, adapter, show_progress = self._build_for_export(output_dir=output)
        resolved_stem = stem or _default_stem(adapter, self._builder, self._report_factory)
        return save_report_bundle(report, output, stem=resolved_stem, show_progress=show_progress)

    def save_html(self, output_path: PathLike) -> Path:
        output = Path(output_path)
        report, _, show_progress = self._build_for_export(output_dir=output.parent)
        return save_html_report(report, output, show_progress=show_progress)

    def save_pdf(self, output_path: PathLike) -> Path:
        output = Path(output_path)
        report, _, show_progress = self._build_for_export(output_dir=output.parent)
        return save_pdf_report(report, output, show_progress=show_progress)

    def _context(self, *, output_dir: Path, assets_dir: Path | None) -> ReportContext:
        return ReportContext(
            model=self._model,
            model_name=self._model_name or type(self._model).__name__,
            X_train=self._X_train,
            X_test=self._X_test,
            y_train=self._y_train,
            y_test=self._y_test,
            predictions=self._predictions,
            feature_names=self._feature_names,
            target_name=self._target_name,
            dataset_source=str(self._options.get("dataset_source", "dataset")),
            output_dir=output_dir,
            assets_dir=assets_dir,
            metadata=self._metadata,
            options=self._options,
        )

    def _build_with_context(
        self,
        *,
        output_dir: Path,
        assets_dir: Path | None,
    ) -> tuple[ExperimentReport, ReportAdapter | None]:
        return _build_report(
            self._context(output_dir=output_dir, assets_dir=assets_dir),
            builder=self._builder,
            report_factory=self._report_factory,
            report_type=self._report_type,
        )

    def _build_for_export(self, *, output_dir: Path) -> tuple[ExperimentReport, ReportAdapter | None, bool]:
        context = self._context(output_dir=output_dir, assets_dir=output_dir / "assets")
        show_progress = bool(self._options.get("show_progress", True))
        _report_progress(
            show_progress,
            f"Building {context.model_name} report data...",
        )
        report, adapter = _build_report(
            context,
            builder=self._builder,
            report_factory=self._report_factory,
            report_type=self._report_type,
        )
        _report_progress(
            show_progress,
            f"Report data ready: {len(report.sections)} sections.",
        )
        return report, adapter, show_progress


def _build_report(
    context: ReportContext,
    *,
    builder: ReportBuilderInput | None,
    report_factory: ReportFactory | None,
    report_type: ReportType,
) -> tuple[ExperimentReport, ReportAdapter | None]:
    if report_factory is not None:
        return _ensure_report(report_factory(context)), None

    if builder is None:
        adapter = resolve_report_adapter(context, report_type)
        if adapter is not None:
            return adapter.build(context), adapter
        return GenericClassificationReportBuilder(context).build(), None

    if isinstance(builder, ExperimentReport):
        return builder, None

    if isinstance(builder, type):
        built = builder(context)
        if isinstance(built, ExperimentReport):
            return built, None
        if hasattr(built, "build"):
            return _ensure_report(built.build()), None
        raise ValueError("Builder class must return an ExperimentReport or an object with build().")

    if hasattr(builder, "build"):
        return _ensure_report(builder.build()), None

    if callable(builder):
        built = builder(context)
        if isinstance(built, ExperimentReport):
            return built, None
        if hasattr(built, "build"):
            return _ensure_report(built.build()), None
        raise ValueError("Builder callable must return an ExperimentReport or an object with build().")

    raise ValueError(
        "builder must be an ExperimentReport, a builder class, an object with build(), "
        "or a callable accepting ReportContext."
    )


def _ensure_report(value: object) -> ExperimentReport:
    if not isinstance(value, ExperimentReport):
        raise ValueError("Custom report builders must produce an ExperimentReport.")
    return value


def _default_stem(
    adapter: ReportAdapter | None,
    builder: ReportBuilderInput | None,
    report_factory: ReportFactory | None,
) -> str:
    if adapter is not None:
        return adapter.default_stem
    if builder is not None or report_factory is not None:
        return "custom_report"
    return "classification_report"


def _report_progress(show_progress: bool, message: str) -> None:
    if show_progress:
        print(f"[ml-method-reports] {message}", file=sys.stderr, flush=True)
