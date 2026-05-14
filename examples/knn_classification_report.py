from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_example_utils import (  # noqa: E402
    DATASET_SOURCE,
    FEATURE_NAMES,
    TARGET_NAME,
    build_classification_dataset,
    log_saved_report,
    split_classification_dataset,
    standard_scale_frames,
)

from ml_method_reports.reporting import report_for  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a KNN nearest-neighbor report.")
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--weights", choices=["uniform", "distance"], default="uniform")
    parser.add_argument("--metric", default="minkowski")
    parser.add_argument("--selected-sample-index", type=int, default=0)
    parser.add_argument(
        "--selected-sample-indices",
        type=str,
        default="",
        help="Comma-separated list of selected test sample indices. Overrides --selected-sample-index when set.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runtime" / "reports")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html_path, pdf_path = create_report_example(
        n_neighbors=args.n_neighbors,
        weights=args.weights,
        metric=args.metric,
        selected_sample_index=args.selected_sample_index,
        selected_sample_indices=_parse_selected_sample_indices(args.selected_sample_indices),
        output_dir=args.output_dir,
        random_state=args.random_state,
    )
    log_saved_report(html_path, pdf_path)


def create_report_example(
    *,
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "minkowski",
    selected_sample_index: int = 0,
    selected_sample_indices: list[int] | None = None,
    output_dir: Path | str = PROJECT_ROOT / "runtime" / "reports",
    random_state: int = 42,
) -> tuple[Path, Path]:
    dataframe = build_classification_dataset(random_state=random_state)
    X_train, X_test, y_train, y_test = split_classification_dataset(
        dataframe, random_state=random_state
    )
    X_train_scaled, X_test_scaled, scaling_params = standard_scale_frames(X_train, X_test)
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
    ).fit(X_train_scaled, y_train)
    return (
        report_for(model)
        .with_training_data(
            X_train=X_train_scaled,
            y_train=y_train,
            feature_names=FEATURE_NAMES,
            target_name=TARGET_NAME,
        )
        .with_test_data(
            X_test=X_test_scaled,
            y_test=y_test,
            feature_names=FEATURE_NAMES,
            target_name=TARGET_NAME,
        )
        .with_options(
            dataset_source=DATASET_SOURCE,
            selected_sample_index=selected_sample_index,
            selected_sample_indices=selected_sample_indices,
            scaling_method="standard",
            scaling_params=scaling_params,
        )
        .save(output_dir, stem="knn_classification_report")
    )


def _parse_selected_sample_indices(value: str) -> list[int] | None:
    if not value.strip():
        return None
    return [int(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
