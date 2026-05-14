from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA

from ml_method_reports.reporting.types import PathLike, TableRows


def project_to_2d(
    X: ArrayLike,
    centers: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    X_array = np.asarray(X, dtype=float)
    if X_array.ndim != 2:
        raise ValueError("X must be a 2D array-like object.")

    centers_array = None if centers is None else np.asarray(centers, dtype=float)
    if centers_array is not None and centers_array.ndim != 2:
        raise ValueError("centers must be a 2D array-like object when provided.")

    feature_count = X_array.shape[1]
    if feature_count == 1:
        X_2d = np.column_stack([X_array[:, 0], np.zeros(X_array.shape[0])])
        centers_2d = None
        if centers_array is not None:
            centers_2d = np.column_stack(
                [centers_array[:, 0], np.zeros(centers_array.shape[0])]
            )
        return X_2d, centers_2d, "single-feature projection with zero y-axis"

    if feature_count == 2:
        return X_array, centers_array, "original 2D features"

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_array)
    centers_2d = None if centers_array is None else pca.transform(centers_array)
    explained = pca.explained_variance_ratio_.sum()
    return X_2d, centers_2d, f"PCA projection to 2D ({explained:.1%} variance)"


def plot_etalon_decision_space(
    X_2d: ArrayLike,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    centers_2d: ArrayLike,
    selected_sample_index: int,
    class_labels: ArrayLike,
    output_path: PathLike,
    title: str = "Etalon Decision Space",
) -> Path:
    plt = _load_pyplot()
    X_array = np.asarray(X_2d, dtype=float)
    centers_array = np.asarray(centers_2d, dtype=float)
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    labels = list(class_labels)

    if selected_sample_index < 0 or selected_sample_index >= X_array.shape[0]:
        raise ValueError("selected_sample_index is outside the available sample range.")

    path = _prepare_output_path(output_path)
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=140)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))

    correct_mask = true_values == predicted_values
    for index, label in enumerate(labels):
        label_mask = true_values == label
        color = colors[index]
        ax.scatter(
            X_array[label_mask & correct_mask, 0],
            X_array[label_mask & correct_mask, 1],
            s=42,
            color=color,
            alpha=0.78,
            label=f"class {label} correct",
        )
        ax.scatter(
            X_array[label_mask & ~correct_mask, 0],
            X_array[label_mask & ~correct_mask, 1],
            s=72,
            color=color,
            marker="o",
            edgecolors="#b91c1c",
            linewidths=1.8,
            label=f"class {label} error",
        )

    for center_index, (label, center) in enumerate(zip(labels, centers_array, strict=True)):
        ax.scatter(
            center[0],
            center[1],
            s=210,
            marker="X",
            color=colors[center_index],
            edgecolors="#111827",
            linewidths=1.2,
            label=f"etalon {label}",
        )

    selected = X_array[selected_sample_index]
    ax.scatter(
        selected[0],
        selected[1],
        s=190,
        marker="*",
        color="#f59e0b",
        edgecolors="#111827",
        linewidths=1.0,
        label="selected sample",
        zorder=5,
    )
    ax.annotate(
        "selected sample",
        xy=(selected[0], selected[1]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        weight="bold",
    )

    for label, center in zip(labels, centers_array, strict=True):
        ax.plot(
            [selected[0], center[0]],
            [selected[1], center[1]],
            linestyle="--",
            linewidth=1.25,
            color="#334155",
            alpha=0.78,
        )
        midpoint = (selected + center) / 2
        distance = np.linalg.norm(selected - center)
        ax.text(midpoint[0], midpoint[1], f"d({label})={distance:.2f}", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("visual axis 1")
    ax.set_ylabel("visual axis 2")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_accuracy_comparison(
    comparison_rows: TableRows,
    output_path: PathLike,
    title: str = "Model Accuracy Comparison",
) -> Path:
    plt = _load_pyplot()
    path = _prepare_output_path(output_path)
    labels = [
        f"{row['model']}\ncustom" if row.get("type") == "custom" else f"{row['model']}\nsklearn"
        for row in comparison_rows
    ]
    values = [float(row["accuracy"]) for row in comparison_rows]
    colors = ["#0f766e" if row.get("type") == "custom" else "#64748b" for row in comparison_rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=140)
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.24)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_full_vs_one_feature_comparison(
    comparison_rows: TableRows,
    output_path: PathLike,
    title: str = "Full-feature vs One-feature Accuracy",
) -> Path:
    plt = _load_pyplot()
    path = _prepare_output_path(output_path)
    labels = [str(row["mode"]).replace(" experiment", "") for row in comparison_rows]
    values = [float(row["accuracy"]) for row in comparison_rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=140)
    bars = ax.bar(labels, values, color=["#0f766e", "#f59e0b"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.24)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_knn_neighbor_space(
    X_train_2d: ArrayLike,
    y_train: ArrayLike,
    selected_sample_2d: ArrayLike,
    neighbor_indices: ArrayLike,
    neighbor_distances: ArrayLike,
    output_path: PathLike,
    title: str = "KNN Nearest Neighbors",
) -> Path:
    plt = _load_pyplot()
    X_array = np.asarray(X_train_2d, dtype=float)
    selected = np.asarray(selected_sample_2d, dtype=float)
    labels = np.asarray(y_train)
    neighbors = np.asarray(neighbor_indices, dtype=int)
    distances = np.asarray(neighbor_distances, dtype=float)

    path = _prepare_output_path(output_path)
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=140)
    unique_labels = np.unique(labels)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_labels), 3)))

    for label_index, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            X_array[mask, 0],
            X_array[mask, 1],
            s=38,
            color=colors[label_index],
            alpha=0.64,
            label=f"class {label}",
        )

    neighbor_points = X_array[neighbors]
    ax.scatter(
        neighbor_points[:, 0],
        neighbor_points[:, 1],
        s=135,
        facecolors="none",
        edgecolors="#111827",
        linewidths=1.8,
        label="nearest neighbors",
        zorder=4,
    )
    ax.scatter(
        selected[0],
        selected[1],
        s=210,
        marker="*",
        color="#f59e0b",
        edgecolors="#111827",
        linewidths=1.0,
        label="selected test sample",
        zorder=5,
    )

    for rank, (point, distance) in enumerate(zip(neighbor_points, distances, strict=True), start=1):
        ax.plot(
            [selected[0], point[0]],
            [selected[1], point[1]],
            linestyle="--",
            linewidth=1.1,
            color="#334155",
            alpha=0.72,
        )
        midpoint = (selected + point) / 2
        ax.text(midpoint[0], midpoint[1], f"#{rank} d={distance:.2f}", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("visual axis 1")
    ax.set_ylabel("visual axis 2")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bar_chart(
    rows: TableRows,
    *,
    label_key: str,
    value_key: str,
    output_path: PathLike,
    title: str,
    ylabel: str = "value",
    color: str = "#0f766e",
) -> Path:
    plt = _load_pyplot()
    path = _prepare_output_path(output_path)
    labels = [str(row[label_key]) for row in rows]
    values = [float(row[value_key]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=140)
    bars = ax.bar(labels, values, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.24)
    ax.tick_params(axis="x", labelrotation=35)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cluster_projection(
    X_2d: ArrayLike,
    labels: ArrayLike,
    output_path: PathLike,
    centers_2d: ArrayLike | None = None,
    title: str = "Cluster Projection",
) -> Path:
    plt = _load_pyplot()
    X_array = np.asarray(X_2d, dtype=float)
    label_values = np.asarray(labels)
    center_array = None if centers_2d is None else np.asarray(centers_2d, dtype=float)
    path = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=140)
    unique_labels = np.unique(label_values)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_labels), 3)))
    for index, label in enumerate(unique_labels):
        mask = label_values == label
        ax.scatter(
            X_array[mask, 0],
            X_array[mask, 1],
            s=42,
            color=colors[index],
            alpha=0.74,
            label=f"cluster {label}",
        )
    if center_array is not None:
        ax.scatter(
            center_array[:, 0],
            center_array[:, 1],
            s=210,
            marker="X",
            color="#f59e0b",
            edgecolors="#111827",
            linewidths=1.1,
            label="centers",
        )
    ax.set_title(title)
    ax.set_xlabel("visual axis 1")
    ax.set_ylabel("visual axis 2")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_agglomerative_dendrogram(
    children: ArrayLike,
    distances: ArrayLike,
    *,
    sample_count: int,
    output_path: PathLike,
    title: str = "Agglomerative Dendrogram",
    truncate_level: int = 5,
) -> Path:
    from scipy.cluster.hierarchy import dendrogram

    plt = _load_pyplot()
    children_array = np.asarray(children, dtype=float)
    distances_array = np.asarray(distances, dtype=float)
    if children_array.ndim != 2 or children_array.shape[1] != 2:
        raise ValueError("children must be a 2D array with two columns.")
    if distances_array.ndim != 1 or distances_array.shape[0] != children_array.shape[0]:
        raise ValueError("distances must contain one value for each merge step.")

    counts = np.zeros(children_array.shape[0], dtype=float)
    for step_index, merge in enumerate(children_array.astype(int)):
        current_count = 0.0
        for child_index in merge:
            if child_index < sample_count:
                current_count += 1.0
            else:
                current_count += counts[child_index - sample_count]
        counts[step_index] = current_count

    linkage_matrix = np.column_stack([children_array, distances_array, counts]).astype(float)
    path = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(10.0, 5.6), dpi=140)
    dendrogram(
        linkage_matrix,
        truncate_mode="level",
        p=truncate_level,
        ax=ax,
        leaf_rotation=45,
        leaf_font_size=8,
    )
    ax.set_title(title)
    ax.set_xlabel("sample index or merged cluster")
    ax.set_ylabel("distance")
    ax.grid(axis="y", alpha=0.24)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _prepare_output_path(output_path: PathLike) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt

