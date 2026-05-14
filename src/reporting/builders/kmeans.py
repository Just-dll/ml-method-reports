from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_method_reports.reporting.builders.base import ReportAssetBuilder, ReportBuilder
from ml_method_reports.reporting.builders.sklearn_common import (
    as_2d_array,
    build_bar_asset,
    class_distribution_rows,
    feature_names,
    model_params_rows,
    preprocessing_section,
    selected_index,
    selected_query,
)
from ml_method_reports.reporting.models import ExperimentReport, ReportSection
from ml_method_reports.reporting.plots import plot_cluster_projection, project_to_2d
from ml_method_reports.reporting.types import FeatureMatrix, ScalingParams, TableRows, TargetVector
from sklearn.cluster import KMeans


@dataclass(slots=True)
class KMeansReportInput:
    model: KMeans
    X: FeatureMatrix
    feature_names: list[str] | None = None
    true_labels: TargetVector | None = None
    dataset_source: str = "clustering dataset"
    selected_sample_index: int = 0
    scaling_method: str = "none"
    scaling_params: ScalingParams | None = None


class KMeansReportBuilder(ReportBuilder, ReportAssetBuilder):
    def __init__(self, report_input: KMeansReportInput, assets_dir: Path | None = None) -> None:
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
        X_2d, centers_2d, projection_info = project_to_2d(X, self._input.model.cluster_centers_)
        path = plot_cluster_projection(X_2d, labels, assets_dir / "kmeans_clusters.png", centers_2d=centers_2d, title="KMeans Cluster Projection")
        assets = {"clusters": str(Path("assets") / path.name), "projection_info": projection_info}
        assets.update(build_bar_asset(class_distribution_rows(labels, "cluster"), assets_dir, "kmeans_cluster_sizes.png", label_key="cluster", value_key="count", title="KMeans Cluster Sizes"))
        return assets

    def _build_report(self, assets: dict[str, str]) -> ExperimentReport:
        data = self._input
        X = as_2d_array(data.X)
        names = feature_names(data.X, data.feature_names)
        labels = np.asarray(data.model.labels_)
        index = selected_index(data.selected_sample_index, X.shape[0])
        distances = data.model.transform(selected_query(data.X, X, index))[0]

        sections = [
            ReportSection(title="1. Experiment Overview", table=[
                {"item": "dataset source", "value": data.dataset_source},
                {"item": "samples", "value": int(X.shape[0])},
                {"item": "features", "value": int(X.shape[1])},
                {"item": "clusters", "value": int(data.model.n_clusters)},
                {"item": "selected sample index", "value": index},
            ]),
            preprocessing_section(data.scaling_method, data.scaling_params, names),
            ReportSection(title="3. KMeans Parameters", table=model_params_rows(data.model, ["n_clusters", "init", "n_init", "max_iter", "random_state"])),
            ReportSection(title="4. Cluster Centers", content="Cluster centers are centroids in feature space. They are not class prototypes unless external labels are separately supplied.", table=_center_rows(data.model, names)),
            ReportSection(title="5. Cluster Size Summary", table=class_distribution_rows(labels, "cluster"), image_path=assets.get("kmeans_cluster_sizes")),
            ReportSection(title="6. Selected Sample Cluster Assignment", content=f"Selected sample {index} is assigned to cluster {int(labels[index])}."),
            ReportSection(title="7. Distances to Centers", table=[{"cluster": cluster, "distance": float(distance)} for cluster, distance in enumerate(distances)]),
            ReportSection(title="8. Inertia / Iteration Summary", table=[
                {"metric": "inertia", "value": float(data.model.inertia_)},
                {"metric": "n_iter", "value": int(data.model.n_iter_)},
            ]),
            ReportSection(title="9. Cluster Visualization", content=f"{assets.get('projection_info', 'Visualization not generated')}. Cluster labels are arbitrary identifiers.", image_path=assets.get("clusters")),
            ReportSection(title="10. Optional External Label Comparison", content=_external_label_note(data.true_labels)),
            ReportSection(title="11. Analysis Summary", content=f"KMeans found {data.model.n_clusters} clusters with inertia {data.model.inertia_:.3f}. This is unsupervised clustering, so no true accuracy is reported unless external labels are supplied only for comparison."),
        ]
        return ExperimentReport(
            title="KMeans Educational Report",
            subtitle="Centroid-based clustering explanation using sklearn KMeans artifacts.",
            metadata={"model": "KMeans", "clusters": int(data.model.n_clusters), "inertia": float(data.model.inertia_)},
            sections=sections,
        )


def _center_rows(model: KMeans, names: list[str]) -> TableRows:
    return [
        {"cluster": cluster, "feature": name, "center_value": float(value)}
        for cluster, center in enumerate(model.cluster_centers_)
        for name, value in zip(names, center, strict=True)
    ]


def _external_label_note(labels: TargetVector | None) -> str:
    if labels is None:
        return "No external labels were supplied. This report does not compute supervised accuracy for clustering."
    return "External labels were supplied and may be used for separate comparison, but they are not training targets for KMeans."

