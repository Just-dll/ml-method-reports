from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ml_method_reports.reporting.builders.base import ReportAssetBuilder, ReportBuilder
from ml_method_reports.reporting.builders.sklearn_common import (
    as_2d_array,
    build_bar_asset,
    class_distribution_rows,
    feature_names,
    model_params_rows,
    preprocessing_section,
)
from ml_method_reports.reporting.models import ExperimentReport, ReportSection
from ml_method_reports.reporting.plots import (
    plot_agglomerative_dendrogram,
    plot_cluster_projection,
    project_to_2d,
)
from ml_method_reports.reporting.types import FeatureMatrix, ScalingParams, TableRows, TargetVector


@dataclass(slots=True)
class AgglomerativeReportInput:
    model: AgglomerativeClustering
    X: FeatureMatrix
    feature_names: list[str] | None = None
    true_labels: TargetVector | None = None
    dataset_source: str = "clustering dataset"
    scaling_method: str = "none"
    scaling_params: ScalingParams | None = None


class AgglomerativeReportBuilder(ReportBuilder, ReportAssetBuilder):
    def __init__(self, report_input: AgglomerativeReportInput, assets_dir: Path | None = None) -> None:
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
        X = as_2d_array(self._input.X)
        labels = np.asarray(self._input.model.labels_)
        X_2d, _, projection_info = project_to_2d(X)
        path = plot_cluster_projection(X_2d, labels, assets_dir / "agglomerative_clusters.png", title="Agglomerative Cluster Projection")
        assets = {"clusters": str(Path("assets") / path.name), "projection_info": projection_info}
        assets.update(build_bar_asset(class_distribution_rows(labels, "cluster"), assets_dir, "agglomerative_cluster_sizes.png", label_key="cluster", value_key="count", title="Agglomerative Cluster Sizes"))
        distances = getattr(self._input.model, "distances_", None)
        children = getattr(self._input.model, "children_", None)
        if distances is not None and children is not None:
            dendrogram_path = plot_agglomerative_dendrogram(
                children,
                distances,
                sample_count=X.shape[0],
                output_path=assets_dir / "agglomerative_dendrogram.png",
            )
            assets["dendrogram"] = str(Path("assets") / dendrogram_path.name)
        return assets

    def _build_report(self, assets: dict[str, str]) -> ExperimentReport:
        data = self._input
        X = as_2d_array(data.X)
        names = feature_names(data.X, data.feature_names)
        labels = np.asarray(data.model.labels_)

        sections = [
            ReportSection(title="1. Experiment Overview", table=[
                {"item": "dataset source", "value": data.dataset_source},
                {"item": "samples", "value": int(X.shape[0])},
                {"item": "features", "value": int(X.shape[1])},
                {"item": "clusters", "value": int(getattr(data.model, "n_clusters_", len(np.unique(labels))))},
            ]),
            preprocessing_section(data.scaling_method, data.scaling_params, names),
            ReportSection(title="3. Agglomerative Parameters", table=model_params_rows(data.model, ["n_clusters", "metric", "linkage", "compute_distances"])),
            ReportSection(title="4. Cluster Size Summary", table=class_distribution_rows(labels, "cluster"), image_path=assets.get("agglomerative_cluster_sizes")),
            ReportSection(title="5. Merge Tree Summary", content="Agglomerative clustering builds clusters by merging samples or groups step by step.", table=_merge_rows(data.model)),
            ReportSection(title="6. Cluster Visualization", content=f"{assets.get('projection_info', 'Visualization not generated')}. Cluster labels are arbitrary identifiers.", image_path=assets.get("clusters")),
            ReportSection(title="7. Dendrogram", content=_dendrogram_note(data.model, bool(assets.get("dendrogram"))), image_path=assets.get("dendrogram")),
            ReportSection(title="8. Optional External Label Comparison", content=_external_label_note(data.true_labels)),
            ReportSection(title="9. Limitations for New Samples", content="AgglomerativeClustering has no native predict() method for assigning future samples. New-sample assignment needs a separate approximation."),
            ReportSection(title="10. Analysis Summary", content=f"AgglomerativeClustering produced {len(np.unique(labels))} cluster(s). This is unsupervised, so no supervised accuracy is reported by default."),
        ]
        return ExperimentReport(
            title="Agglomerative Clustering Educational Report",
            subtitle="Hierarchical clustering explanation using sklearn AgglomerativeClustering artifacts.",
            metadata={"model": "AgglomerativeClustering", "clusters": int(len(np.unique(labels)))},
            sections=sections,
        )


def _merge_rows(model: AgglomerativeClustering) -> TableRows:
    children = getattr(model, "children_", np.empty((0, 2), dtype=int))
    distances = getattr(model, "distances_", None)
    rows = []
    for step, pair in enumerate(children[:12], start=1):
        row = {"merge_step": step, "left": int(pair[0]), "right": int(pair[1])}
        if distances is not None:
            row["distance"] = float(distances[step - 1])
        rows.append(row)
    return rows


def _dendrogram_note(model: AgglomerativeClustering, generated: bool) -> str:
    if generated:
        return "The dendrogram shows the hierarchy of cluster merges derived from children_ and distances_. For readability, the plot is truncated to the last hierarchy levels."
    if hasattr(model, "distances_"):
        return "Distances are available, but the dendrogram image was not generated because assets were not requested for this report build."
    return "No distances_ attribute is available. Fit with compute_distances=True to include a dendrogram in the saved report."


def _external_label_note(labels: TargetVector | None) -> str:
    if labels is None:
        return "No external labels were supplied. Cluster IDs are not class labels."
    return "External labels were supplied for optional comparison only; they were not used as clustering targets."

